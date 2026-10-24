"""Flash-style attention with a per-key additive bias (Triton).

F.scaled_dot_product_attention cannot use the flash kernel when attn_mask is
given, so the bcastq_prop proportional-attention bias (log token size,
[B, H, 1, N_kv]) forces the slower mem-efficient backend. The bias depends only
on the key index, so a flash-style online-softmax kernel can add it to each
score tile for free and keep close to flash speed.

Math matches SDPA(q, k, v, attn_mask=bias): scores = q@k^T * scale + bias,
softmax over keys in fp32, P@V accumulated in fp32 (P cast to input dtype for
the tensor-core matmul, as in FlashAttention).

Structure follows FlashAttention-2 rather than the original online-softmax form:
  * the P@V accumulate is folded into the MMA via the three-operand tl.dot
  * the inner loop carries no masks at all -- key indices past N are clamped to
    N-1 (so every load is in bounds and needs no predicate) and the bias is -1e9
    there, so those duplicate keys are killed by the softmax. Clamping rather
    than zero-padding K/V avoids allocating padded copies, which at S=200 cost
    ~11 GB of peak memory.
  * the bias arrives already multiplied by log2(e) in fp32, so the loop adds it
    straight to the exp2-domain scores

Measured at the merged-block shape (M=275k, N=88k, H=16, D=64) on an RTX A6000:
1138 ms for the previous kernel, 972 ms here, against a 935 ms flash-without-bias
floor -- about 82% of the gap closed, at identical accuracy (1.2e-4 vs SDPA).

Tried and rejected, so deliberately absent:
  * fp16 MMA accumulate (GA102 runs it at 2x the fp32-accumulate rate): 0.6%
    faster, 5x the error. The kernel sits at ~64% of fp16 peak, so it is bound
    by softmax/scheduling overhead rather than MMA throughput.
  * pre-scaling q by sm_scale*log2(e) on the host: both slower (an extra
    full-tensor fp32 pass over q) and less accurate (4.9e-4 vs 1.2e-4) than
    scaling the fp32 scores inside the loop.
  * materialising K^T, or loading K^T with swapped strides: no gain either way.
"""
import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634
_PAD = 128          # must be >= the largest BLOCK_N below


_CONFIGS = [
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 64}, num_warps=8, num_stages=3),
]
# Deliberately absent: {"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=4.
# It compiles to 108032 B of shared memory, above the 101376 B opt-in limit of the
# card this is developed on (RTX A6000, sm_86), so every launch raised
# OutOfResources from triton/compiler/compiler.py::_init_handles. That is not just
# a wasted autotune slot: the raise happens at *launch*, and the exception's
# traceback keeps every frame of the launching call chain alive -- Triton's own
# comment at that line warns it "will keep references to all the locals in the
# traceback". A frame a traceback points at cannot be freed by refcounting, so the
# K/V and DPT taps of the forward that triggered autotuning stayed resident (10.6
# GiB at 200 frames) until the cyclic collector happened to run. Reinstate it only
# behind a shared-memory check, on a card with a large enough limit.

# Keyed on the full shape, not just HEAD_DIM. Several call sites share D=64 but
# have very different (M, N): the merged global blocks (M~275k, N~88k), the shared
# broadcast attention (M~275k, N~6k) and the frame-local one (M~1.4k, N~1.4k,
# batch 3200). Keying on D alone let whichever shape ran first pin the tile for
# all of them, costing ~20% end to end. Re-tuning per shape costs a few seconds
# once per process. warmup/rep are above the defaults so that two near-equal
# configs do not swap between runs (which would change fp16 accumulation order).
@triton.autotune(configs=_CONFIGS, key=["M", "Npad", "HEAD_DIM"], warmup=50, rep=300)
@triton.jit
def _attn_keybias_kernel(
    q_ptr, k_ptr, v_ptr, bias_ptr, o_ptr, lse_ptr,
    M, N, Npad,
    stride_qb, stride_qm,
    stride_kb, stride_kn,
    stride_vb, stride_vn,
    stride_bb,
    stride_ob, stride_om,
    stride_lb,
    qk_scale,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    RETURN_LSE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    m_mask = offs_m < M

    q = tl.load(q_ptr + pid_b * stride_qb + offs_m[:, None] * stride_qm + offs_d[None, :],
                mask=m_mask[:, None], other=0.0)

    m_i = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)

    kb = k_ptr + pid_b * stride_kb
    vb = v_ptr + pid_b * stride_vb
    bb = bias_ptr + pid_b * stride_bb

    for n_start in range(0, Npad, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        offs_kv = tl.minimum(offs_n, N - 1)     # in-bounds; bias kills the dupes
        k = tl.load(kb + offs_kv[:, None] * stride_kn + offs_d[None, :])
        qk = tl.dot(q, tl.trans(k)) * qk_scale + tl.load(bb + offs_n)[None, :]

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, 1)

        v = tl.load(vb + offs_kv[:, None] * stride_vn + offs_d[None, :])
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(v.dtype), v, acc)
        m_i = m_ij

    out = acc / l_i[:, None]
    tl.store(o_ptr + pid_b * stride_ob + offs_m[:, None] * stride_om + offs_d[None, :],
             out.to(o_ptr.dtype.element_ty), mask=m_mask[:, None])
    if RETURN_LSE:
        # log2 of this partition's softmax denominator, so two partitions of the
        # key set can be merged exactly:  out = (oA*2^(lA-m) + oB*2^(lB-m)) / (...)
        tl.store(lse_ptr + pid_b * stride_lb + offs_m,
                 m_i + tl.math.log2(l_i), mask=m_mask)


def attention_keybias(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                      bias: torch.Tensor, sm_scale: float, return_lse: bool = False):
    """SDPA(q, k, v, attn_mask=bias) where bias is per-key: [B, H, 1, N].

    q [B, H, M, D], k/v [B, H, N, D] (CUDA, fp16/bf16). Returns [B, H, M, D].
    With return_lse, also returns [B, H, M] fp32 log2 softmax denominators, which
    let a caller split the key set across several calls and combine exactly.
    `bias` may be None (plain attention).
    """
    if q.dtype == torch.float32:
        # mimic SDPA-under-autocast: it is autocast-aware and would have received
        # bf16-cast inputs (q/k are fp32 here because qk_norm runs in fp32 there).
        dt = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else torch.bfloat16
        q, k, v = q.to(dt), k.to(dt), v.to(dt)
        bias = None if bias is None else bias.to(dt)

    B, H, M, D = q.shape
    N = k.shape[2]
    BH = B * H
    Npad = triton.cdiv(N, _PAD) * _PAD

    # The Triton kernel already receives explicit batch/token strides. Avoid
    # forcing a contiguous [B*H, N, D] pack when the head-major inputs are
    # valid strided views of QKV storage; at long sequences that needlessly
    # overlaps three full Q/K/V copies with the caller-owned tensors.
    qc = q.reshape(BH, M, D)
    kp = k.reshape(BH, N, D)
    vp = v.to(q.dtype).reshape(BH, N, D)
    # only the bias is padded (a few hundred KB); K/V are read with clamped indices
    bp = torch.full((BH, Npad), -1e9, device=q.device, dtype=torch.float32)
    bp[:, :N] = 0.0 if bias is None else bias.reshape(BH, N).float() * LOG2E

    o = torch.empty_like(qc)
    lse = (torch.empty(BH, M, device=q.device, dtype=torch.float32)
           if return_lse else torch.empty(1, device=q.device, dtype=torch.float32))

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), BH)
    _attn_keybias_kernel[grid](
        qc, kp, vp, bp, o, lse,
        M, N, Npad,
        qc.stride(0), qc.stride(1),
        kp.stride(0), kp.stride(1),
        vp.stride(0), vp.stride(1),
        bp.stride(0),
        o.stride(0), o.stride(1),
        lse.stride(0) if return_lse else 0,
        sm_scale * LOG2E,
        HEAD_DIM=D, RETURN_LSE=return_lse,
    )
    o = o.reshape(B, H, M, D)
    return (o, lse.reshape(B, H, M)) if return_lse else o
