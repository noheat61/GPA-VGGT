import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

from merging.merge import token_merge_bipartite2d

try:  # flash-style attention with a per-key additive bias (Triton). PyTorch SDPA
    # falls back to a slower backend whenever attn_mask is set.
    from merging.fused_attention import attention_keybias
except ImportError:
    attention_keybias = None

XFORMERS_AVAILABLE = False


class Attention(nn.Module):
    """VGGT attention with the two GPA-VGGT global-attention paths.

    Set by the Aggregator on the global blocks:

    * ``frame_special`` (early blocks): frame attention plus a compact
      cross-frame K/V set shared by all frames -- every frame's special tokens
      and ``broadcast_pool``-strided, content-adaptive pooled patch summaries.
    * ``global_merging`` passed to ``forward`` (later blocks): query-preserving
      K/V merging with a ``+log s`` correction on each merged key.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,
        rope=None,
        patch_width: int = 37,
        patch_height: int = 28,
        merge_ratio: float = 0.9,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.patch_width = patch_width
        self.patch_height = patch_height
        self.fused_attn = fused_attn
        self.merge_ratio = merge_ratio

        # Early global blocks: frame attention + shared cross-frame K/V set.
        self.frame_special = False
        self.num_special = 5  # 1 camera + 4 register tokens per frame
        # Seed stride of the pooled summaries (0 = special tokens only).
        self.broadcast_pool = 0
        # dst grid stride of the bipartite merge: one dst per 2x2 window.
        self.merge_grid = 2
        # Matching runs on a random projection to this many dims (0 = full dim).
        self.merge_metric_dim = 0

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def _frame_special_attention(self, q, k, v, B, N, C):
        """Frame attention with a shared cross-frame K/V set.

            queries   : every token of frame s
            keys/vals : own patch tokens of frame s
                     ++ camera + register tokens of every frame
                     ++ pooled patch summaries of every frame

        The groups are disjoint, so no key is duplicated and the softmax is exact.
        Each pooled summary stands for ``s`` patch tokens and gets a ``+log s``
        logit bias. RoPE is already applied to q/k by the caller.
        """
        # q_norm/k_norm are LayerNorms, which autocast promotes to fp32; the
        # 16-bit kernels below need q/k/v in the autocast dtype.
        if q.dtype == torch.float32 and torch.is_autocast_enabled():
            dt = torch.get_autocast_gpu_dtype()
            q, k, v = q.to(dt), k.to(dt), v.to(dt)

        H, D = self.num_heads, self.head_dim
        pts = self.num_special
        pw, ph = self.patch_width, self.patch_height
        P = pw * ph + pts
        S = N // P
        if S * P != N:
            raise ValueError(
                f"frame_special: token count {N} is not S*(patch_w*patch_h+{pts}) "
                f"with patch grid {self.patch_width}x{self.patch_height} -- did "
                "update_patch_dimensions() run for this input?"
            )

        # queries: all P tokens of each frame, frames folded into the batch dim
        q_f = q.view(B, H, S, P, D).permute(0, 2, 1, 3, 4).reshape(B * S, H, P, D)

        kp = self.broadcast_pool
        nh, nw = (-(-ph // kp), -(-pw // kp)) if kp > 0 else (0, 0)
        npool = nh * nw
        pool_assign = None
        pool_size = None
        acc_dtype = torch.float64 if q_f.dtype == torch.float64 else torch.float32

        def _pool(t5):
            """Patch tokens -> npool content-adaptive summaries per frame.

            Seeds are the stride-kp lattice tokens; every patch joins the seed
            with the highest cosine similarity of its head-h key, and a summary
            is the mean over its group. The grouping is decided on the keys and
            reused for the values. Returns [B, H, S*npool, D].
            """
            nonlocal pool_assign, pool_size
            x = t5[:, :, :, pts:, :].reshape(B, H, S, ph, pw, D)
            xf = x.reshape(B, H, S, ph * pw, D)
            if pool_assign is None:
                dst = x[:, :, :, ::kp, ::kp, :].reshape(B, H, S, npool, D)
                a = xf.float()
                b = dst.float()
                a = a / a.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                b = b / b.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                pool_assign = torch.einsum("...id,...gd->...ig", a, b).argmax(-1).contiguous()
                del a, b
                pool_size = torch.zeros(B, H, S, npool, device=x.device, dtype=acc_dtype)
                pool_size.scatter_add_(3, pool_assign,
                                       torch.ones_like(pool_assign, dtype=acc_dtype))
            acc = torch.zeros(B, H, S, npool, D, device=x.device, dtype=acc_dtype)
            acc.scatter_add_(3, pool_assign[..., None].expand(B, H, S, ph * pw, D),
                             xf.to(acc_dtype))
            # a seed is its own group's best match, so no group is empty
            acc /= pool_size.clamp_min(1.0).unsqueeze(-1)
            return acc.reshape(B, H, S * npool, D).to(x.dtype)

        def _pooled_bias(ntok, n_head):
            """[B, H, 1, ntok] additive bias: +log s on the pooled keys, 0 elsewhere."""
            b = torch.zeros(B, H, 1, ntok, device=q.device, dtype=acc_dtype)
            b[:, :, 0, n_head:] = pool_size.log().reshape(B, H, -1)
            return b

        # ---- split path: keep the shared K/V shared instead of replicating it ----
        # Attend to the own-frame keys and to the shared keys separately and
        # combine the two with their log-sum-exp, which is exact. The fallback
        # below is the same math with a per-frame copy of the shared set.
        if (kp > 0 and attention_keybias is not None and q_f.is_cuda
                and not self.training and q_f.dtype in (torch.float16, torch.bfloat16)):
            k5, v5 = k.view(B, H, S, P, D), v.view(B, H, S, P, D)
            own_k = k5[:, :, :, pts:, :].permute(0, 2, 1, 3, 4).reshape(B * S, H, P - pts, D)
            own_v = v5[:, :, :, pts:, :].permute(0, 2, 1, 3, 4).reshape(B * S, H, P - pts, D)
            out_a, lse_a = attention_keybias(q_f, own_k, own_v, None, self.scale, return_lse=True)

            sh_k = torch.cat([k5[:, :, :, :pts, :].reshape(B, H, S * pts, D), _pool(k5)], dim=2)
            sh_v = torch.cat([v5[:, :, :, :pts, :].reshape(B, H, S * pts, D), _pool(v5)], dim=2)
            sh_bias = _pooled_bias(sh_k.shape[2], S * pts).to(q_f.dtype).contiguous()
            out_b, lse_b = attention_keybias(q, sh_k, sh_v, sh_bias, self.scale, return_lse=True)

            out_a = out_a.reshape(B, S, H, P, D).permute(0, 2, 1, 3, 4).reshape(B, H, N, D)
            lse_a = lse_a.reshape(B, S, H, P).permute(0, 2, 1, 3).reshape(B, H, N)
            mx = torch.maximum(lse_a, lse_b)
            wa, wb = torch.exp2(lse_a - mx), torch.exp2(lse_b - mx)
            den = wa + wb
            # normalise the weights, then combine in place in the input dtype
            ca = (wa / den).to(q.dtype).unsqueeze(-1)
            cb = (wb / den).to(q.dtype).unsqueeze(-1)
            del mx, wa, wb, den, lse_a, lse_b
            out_a.mul_(ca)
            out_a.addcmul_(out_b, cb)
            del out_b, ca, cb
            return out_a.transpose(1, 2).reshape(B, N, C)

        def _kv(t):
            t5 = t.view(B, H, S, P, D)
            if kp < 0:
                # no shared set: plain frame attention
                return t5.permute(0, 2, 1, 3, 4).reshape(B * S, H, P, D)
            own = t5[:, :, :, pts:, :].permute(0, 2, 1, 3, 4).reshape(B * S, H, P - pts, D)
            sp = t5[:, :, :, :pts, :].reshape(B, H, S * pts, D)
            sp = sp.unsqueeze(1).expand(B, S, H, S * pts, D).reshape(B * S, H, S * pts, D)
            if kp == 0:
                return torch.cat([own, sp], dim=2)
            pooled = _pool(t5).unsqueeze(1).expand(B, S, H, -1, D).reshape(B * S, H, -1, D)
            return torch.cat([own, sp, pooled], dim=2)

        k_f, v_f = _kv(k), _kv(v)
        bias = None
        if kp > 0:
            ntok = k_f.shape[2]
            bias = (_pooled_bias(ntok, ntok - S * npool).unsqueeze(1)
                    .expand(B, S, H, 1, ntok).reshape(B * S, H, 1, ntok)
                    .to(q.dtype).contiguous())

        if (bias is not None and attention_keybias is not None and q.is_cuda
                and not self.training
                and q.dtype in (torch.float16, torch.bfloat16, torch.float32)):
            out = attention_keybias(q_f, k_f, v_f, bias, self.scale)
        else:
            out = F.scaled_dot_product_attention(q_f, k_f, v_f, attn_mask=bias)
        out = out.reshape(B, S, H, P, D).permute(0, 2, 1, 3, 4).reshape(B, H, N, D)
        return out.transpose(1, 2).reshape(B, N, C)

    def forward(self, x: Tensor, pos=None, global_merging=None) -> Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        if self.frame_special:
            # v is still a view into the 3C qkv projection; give it its own
            # storage so the projection can be freed before the attention.
            v = v.contiguous()
            del qkv
            x = self._frame_special_attention(q, k, v, B, N, C)
            del q, k, v
            x = self.proj(x)
            x = self.proj_drop(x)
            return x

        attn_bias = None
        if global_merging is not None:
            # Query-preserving K/V merging: every query keeps its own output, only
            # keys and values are merged, and a merged key standing for s tokens
            # gets +log s on its logit to restore their softmax mass.
            generator = torch.Generator(device=x.device)
            generator.manual_seed(33)
            r = int(x.shape[1] * self.merge_ratio)
            B_q, H_q, N_q, D_q = q.shape

            m = token_merge_bipartite2d(
                x, self.patch_width, self.patch_height,
                self.merge_grid, self.merge_grid, r,
                False, generator,
                metric_dim=self.merge_metric_dim,
                num_special=self.num_special,
            )
            size = m.size
            if size is None:
                ones = torch.ones(B_q, N_q, 1, device=x.device, dtype=torch.float32)
                size = m(ones, mode="sum").reshape(-1)

            k = m.heads(k, mode="mean")
            v = m.heads(v, mode="mean")
            del qkv
            N_m = k.shape[2]
            attn_bias = (size.reshape(B_q, 1, 1, N_m).log().to(q.dtype)
                         .expand(B_q, H_q, 1, N_m).contiguous())

        if (
            attn_bias is not None
            and attention_keybias is not None
            and q.is_cuda
            and not self.training
        ):
            x = attention_keybias(q, k, v, attn_bias, self.scale)
        else:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_bias,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        del q, k, v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    def forward(
        self, x: Tensor, attn_bias=None, pos=None, global_merging=None
    ) -> Tensor:
        assert (
            pos is None or self.rope is not None
        ), "Position encoding is only supported with RoPE"
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x, pos=pos, global_merging=global_merging)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = qkv.unbind(2)

        if self.rope is not None and pos is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        # Use scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim**-0.5)
        if attn_bias is not None:
            attn = attn + attn_bias
        attn = F.softmax(attn, dim=-1)
        x = torch.matmul(attn, v)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
