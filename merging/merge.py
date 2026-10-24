import torch
from typing import Callable, Optional, Tuple

try:  # fused matmul + argmax (Triton)
    from merging.fused_similarity import fused_similarity_argmax
except ImportError:
    fused_similarity_argmax = None

# Seed of the fixed Gaussian basis used by the metric_dim random projection.
PROJ_SEED = 1234


@torch.jit.script
def fast_similarity_chunks(
    a: torch.Tensor, b_transposed: torch.Tensor, chunk_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:

    B, num_src, C = a.shape
    original_dtype = a.dtype

    # Convert to bf16 for computation to improve performance and reduce memory usage
    a_bf16 = a.to(torch.bfloat16)
    b_transposed_bf16 = b_transposed.to(torch.bfloat16)
    node_max = torch.empty(B, num_src, device=a.device, dtype=original_dtype)
    node_idx = torch.empty(B, num_src, device=a.device, dtype=torch.long)

    # Process in chunks
    for i in range(0, num_src, chunk_size):
        end_i = min(i + chunk_size, num_src)
        a_chunk = a_bf16[:, i:end_i, :]  # [B, chunk_size, C]
        scores_chunk = torch.bmm(a_chunk, b_transposed_bf16)
        chunk_max_bf16, chunk_idx = torch.max(scores_chunk, dim=2)
        chunk_max = chunk_max_bf16.to(original_dtype)
        node_max[:, i:end_i] = chunk_max
        node_idx[:, i:end_i] = chunk_idx
    return node_max, node_idx


def token_merge_bipartite2d(
    metric: torch.Tensor,
    w: int,
    h: int,
    sx: int,
    sy: int,
    r: int,
    no_rand: bool = False,
    generator: Optional[torch.Generator] = None,
    metric_dim: int = 0,
    num_special: int = 5,
) -> Callable:
    """Bipartite token merging over the multi-frame token sequence (from FastVGGT).

    Tokens are split into dst (one random token per (sx, sy) window, every token
    of the first frame, and every special token) and src; the r src tokens most
    similar to a dst token are merged into it.

    Args:
     - metric [B, N, C]: features used for matching
     - w, h: image width / height in patch tokens
     - sx, sy: dst stride in x / y
     - r: number of tokens to remove by merging
     - no_rand: use the top-left token of each window as dst instead of a random one
     - generator: RNG for the dst choice
     - metric_dim: match on a fixed Gaussian random projection to this many dims
       (0 = full dim)
     - num_special: non-patch tokens at the head of each frame (never merged)

    Returns:
     - merge(x [B, N, C], mode) -> [B, N - r, C]. ``merge.heads(t [B, H, N, D], mode)``
       merges in head-major layout, and ``merge.size`` holds the number of tokens
       behind each output slot (None for B > 1).
    """
    B, N, _ = metric.shape
    if r <= 0:
        raise ValueError("token_merge_bipartite2d: r must be positive")

    gather = torch.gather

    tokens_per_img = w * h + num_special
    num_imgs = N // tokens_per_img
    assert tokens_per_img * num_imgs == N, (
        f"Token count doesn't match (w*h+{num_special})*num_imgs: "
        f"N={N} w={w} h={h}")

    with torch.no_grad():
        # -1 marks dst, 0 marks src
        idx_buffer_seq = torch.zeros(N, device=metric.device, dtype=torch.int64)
        hsy, wsx = h // sy, w // sx  # Number of blocks within each image

        # Mark first image entirely as dst
        if num_imgs > 0:
            idx_buffer_seq[:tokens_per_img] = -1

        if num_imgs > 1:
            cls_indices = (
                torch.arange(1, num_imgs, device=metric.device) * tokens_per_img
            )
            cls_indices = cls_indices[:, None] + torch.arange(
                num_special, device=metric.device)
            idx_buffer_seq[cls_indices.flatten()] = -1
            effective_h = min(hsy * sy, h)
            effective_w = min(wsx * sx, w)
            effective_grid_size = effective_h * effective_w

            if no_rand:
                base_pattern = torch.zeros(
                    effective_grid_size, device=metric.device, dtype=torch.int64
                )
                grid_starts = (
                    torch.arange(1, num_imgs, device=metric.device) * tokens_per_img
                    + num_special
                )
                grid_indices = grid_starts[:, None] + torch.arange(
                    effective_grid_size, device=metric.device
                )
                idx_buffer_seq[grid_indices.flatten()] = base_pattern.repeat(
                    num_imgs - 1
                )
            else:
                total_other_imgs = num_imgs - 1
                all_rand_idx = torch.randint(
                    sy * sx,
                    size=(total_other_imgs, hsy, wsx),
                    device=metric.device,
                    generator=generator,
                )

                scatter_src = -torch.ones(
                    total_other_imgs, hsy, wsx, device=metric.device, dtype=torch.int64
                )

                idx_buffer_batch = torch.zeros(
                    total_other_imgs,
                    hsy,
                    wsx,
                    sy * sx,
                    device=metric.device,
                    dtype=torch.int64,
                )
                idx_buffer_batch.scatter_(
                    dim=3,
                    index=all_rand_idx.unsqueeze(-1),
                    src=scatter_src.unsqueeze(-1),
                )

                idx_buffer_batch = (
                    idx_buffer_batch.view(total_other_imgs, hsy, wsx, sy, sx)
                    .transpose(2, 3)
                    .reshape(total_other_imgs, hsy * sy, wsx * sx)
                )

                # one scatter for all frames
                flat = idx_buffer_batch[:, :effective_h, :effective_w].reshape(
                    total_other_imgs, -1
                )
                grid_starts = (
                    torch.arange(1, num_imgs, device=metric.device) * tokens_per_img
                    + num_special
                )
                write_idx = grid_starts[:, None] + torch.arange(
                    effective_grid_size, device=metric.device
                )
                idx_buffer_seq[write_idx.reshape(-1)] = flat.reshape(-1)

        rand_idx = idx_buffer_seq.reshape(1, -1, 1).argsort(dim=1)
        num_dst = int((idx_buffer_seq == -1).sum())

        a_idx = rand_idx[:, num_dst:, :]  # src
        b_idx = rand_idx[:, :num_dst, :]  # dst
        num_src = a_idx.shape[1]

        def split(x):
            C = x.shape[-1]
            src = gather(x, dim=1, index=a_idx.expand(B, num_src, C))
            dst = gather(x, dim=1, index=b_idx.expand(B, num_dst, C))
            return src, dst

        # Cosine similarity. A Gaussian random projection to metric_dim preserves
        # cosine similarity and makes the matching ~C/metric_dim times cheaper.
        if metric_dim and metric_dim < metric.shape[-1]:
            g_proj = torch.Generator(device=metric.device)
            g_proj.manual_seed(PROJ_SEED)
            proj = torch.randn(metric.shape[-1], metric_dim, device=metric.device,
                               dtype=torch.float32, generator=g_proj).to(metric.dtype)
            metric = metric @ proj
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = split(metric)

        r = min(a.shape[1], r)
        num_src_actual = a.shape[1]
        chunk_size = min(5000, num_src_actual)

        if fused_similarity_argmax is not None and a.is_cuda:
            # keeps each score tile in registers instead of materialising the
            # whole [num_src, num_dst] score matrix
            node_max, node_idx = fused_similarity_argmax(a, b)
        else:
            node_max, node_idx = fast_similarity_chunks(a, b.transpose(-1, -2), chunk_size)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        unm_idx = edge_idx[..., r:, :]
        src_idx = edge_idx[..., :r, :]

        # dst slot of each merged src token, and absolute input positions so the
        # merge can gather directly from its input
        dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)
        unm_abs_idx = gather(a_idx.expand(B, -1, -1), dim=-2, index=unm_idx)
        src_abs_idx = gather(a_idx.expand(B, -1, -1), dim=-2, index=src_idx)

        # number of input tokens behind each output slot
        merged_size = torch.ones(
            B, unm_idx.shape[1] + num_dst, device=metric.device, dtype=torch.float32,
        )
        merged_size[:, unm_idx.shape[1]:].scatter_add_(
            1, dst_idx.squeeze(-1),
            torch.ones(B, r, device=metric.device, dtype=torch.float32),
        )

    unm_len = unm_idx.shape[1]
    src_len = src_idx.shape[1]

    def merge(x: torch.Tensor, mode: str = "mean") -> torch.Tensor:
        """[B, N, C] -> [B, unm_len + num_dst, C]: unmerged src, then dst."""
        n_t, _, c_t = x.shape
        out = torch.empty(n_t, unm_len + num_dst, c_t, device=x.device, dtype=x.dtype)
        head = out[:, :unm_len]
        mid = out[:, unm_len:]
        torch.gather(x, 1, unm_abs_idx.expand(n_t, unm_len, c_t), out=head)
        torch.gather(x, 1, b_idx.expand(n_t, num_dst, c_t), out=mid)
        src_r = gather(x, 1, src_abs_idx.expand(n_t, src_len, c_t))
        mid.scatter_reduce_(1, dst_idx.expand(n_t, src_len, c_t), src_r, reduce=mode)
        return out

    def merge_heads(t: torch.Tensor, mode: str = "mean") -> torch.Tensor:
        """Merge [B, H, N, D] K/V directly in the attention-kernel layout."""
        n_t, h_t, n_tokens, d_t = t.shape
        if n_t != B or n_tokens != N:
            raise ValueError("merge_heads input does not match the merge mapping")
        out = torch.empty(n_t, h_t, unm_len + num_dst, d_t, device=t.device, dtype=t.dtype)
        head = out[:, :, :unm_len]
        mid = out[:, :, unm_len:]
        unm_i = unm_abs_idx.squeeze(-1)[:, None, :, None].expand(n_t, h_t, unm_len, d_t)
        dst_i = b_idx.squeeze(-1)[:, None, :, None].expand(n_t, h_t, num_dst, d_t)
        src_i = src_abs_idx.squeeze(-1)[:, None, :, None].expand(n_t, h_t, src_len, d_t)
        dst_r = dst_idx.squeeze(-1)[:, None, :, None].expand(n_t, h_t, src_len, d_t)
        torch.gather(t, 2, unm_i, out=head)
        torch.gather(t, 2, dst_i, out=mid)
        src_r = gather(t, 2, src_i)
        mid.scatter_reduce_(2, dst_r, src_r, reduce=mode)
        return out

    # The source ordering is data-dependent per batch element, so the shared
    # slot sizes are only valid for B == 1; callers fall back to merge(ones, "sum").
    merge.size = merged_size if B == 1 else None
    merge.heads = merge_heads
    return merge
