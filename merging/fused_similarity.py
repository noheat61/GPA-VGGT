"""Fused similarity + argmax Triton kernel for bipartite token matching.

Computes node_max, node_idx = (a @ b^T).max(dim=-1) without materializing the
[B, num_src, num_dst] score matrix. fast_similarity_chunks writes/reads the
full bf16 score chunk to global memory (~1.6 GB traffic per chunk at S=64
head-wise shapes), which makes the matching memory-bound; this kernel keeps
each score tile in registers and only writes the per-row max/argmax.

Numerics: tl.dot accumulates in fp32 (same as cuBLAS bf16 matmul); the max is
taken over fp32 scores and cast back to the input dtype, so results match the
chunked version up to bf16 rounding/tie order.

The reduction dimension is tiled with BLOCK_K. Loading the whole feature dim as
one tile (BLOCK_D = next_power_of_2(D)) is fine for the head-wise path (D = 64)
but asks for 320 KB of shared memory at the head-shared path (D = 1024) and
fails to launch, which is why token_merge_bipartite2d used to fall back to the
chunked bmm there.
"""
import torch
import triton
import triton.language as tl


_CONFIGS = [
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=8, num_stages=3),
]


@triton.autotune(configs=_CONFIGS, key=["M", "N", "D"], warmup=50, rep=300)
@triton.jit
def _sim_argmax_kernel(
    a_ptr, b_ptr, max_ptr, idx_ptr,
    M, N, D,
    stride_ab, stride_am, stride_ad,
    stride_bb, stride_bn, stride_bd,
    stride_xb, stride_xm,
    stride_ib, stride_im,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < M
    a_row = a_ptr + pid_b * stride_ab + offs_m[:, None] * stride_am
    b_base = b_ptr + pid_b * stride_bb

    row_max = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    row_idx = tl.zeros((BLOCK_M,), tl.int32)

    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N
        b_row = b_base + offs_n[:, None] * stride_bn

        scores = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for k_start in range(0, D, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            k_mask = offs_k < D
            a_tile = tl.load(a_row + offs_k[None, :] * stride_ad,
                             mask=m_mask[:, None] & k_mask[None, :], other=0.0)
            b_tile = tl.load(b_row + offs_k[None, :] * stride_bd,
                             mask=n_mask[:, None] & k_mask[None, :], other=0.0)
            scores += tl.dot(a_tile, tl.trans(b_tile))

        scores = tl.where(n_mask[None, :], scores, float("-inf"))
        tile_max = tl.max(scores, axis=1)
        tile_arg = tl.argmax(scores, axis=1)
        update = tile_max > row_max  # strict > keeps the first-seen max on ties
        row_idx = tl.where(update, (n_start + tile_arg).to(tl.int32), row_idx)
        row_max = tl.where(update, tile_max, row_max)

    tl.store(max_ptr + pid_b * stride_xb + offs_m * stride_xm, row_max, mask=m_mask)
    tl.store(idx_ptr + pid_b * stride_ib + offs_m * stride_im, row_idx, mask=m_mask)


def fused_similarity_argmax(a: torch.Tensor, b: torch.Tensor):
    """node_max, node_idx = (a @ b.transpose(-1,-2)).max(dim=-1), fused.

    a: [B, M, D] src metric, b: [B, N, D] dst metric (CUDA, fp16/bf16/fp32).
    Returns node_max [B, M] (a.dtype), node_idx [B, M] (int64) -- same contract
    as fast_similarity_chunks(a, b.transpose(-1, -2), chunk).
    """
    assert a.is_cuda and b.is_cuda and a.dim() == 3 and b.dim() == 3
    out_dtype = a.dtype
    if a.dtype == torch.float32:
        # match fast_similarity_chunks: similarity is computed in bf16 (q/k are
        # fp32 here when qk_norm runs under autocast). Also avoids a Triton 2.3
        # fp32 tl.dot compiler bug ("unsupported layout conversion").
        a = a.to(torch.bfloat16)
        b = b.to(torch.bfloat16)
    B, M, D = a.shape
    N = b.shape[1]
    a = a.contiguous()
    b = b.contiguous()
    node_max = torch.empty(B, M, device=a.device, dtype=torch.float32)
    node_idx = torch.empty(B, M, device=a.device, dtype=torch.int32)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), B)
    _sim_argmax_kernel[grid](
        a, b, node_max, node_idx,
        M, N, D,
        a.stride(0), a.stride(1), a.stride(2),
        b.stride(0), b.stride(1), b.stride(2),
        node_max.stride(0), node_max.stride(1),
        node_idx.stride(0), node_idx.stride(1),
    )
    return node_max.to(out_dtype), node_idx.to(torch.long)
