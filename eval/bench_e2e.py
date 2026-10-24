"""End-to-end latency of the VGGT forward pass (aggregator + camera and depth heads).

Run one configuration per process: Triton autotuning and the allocator cache
persist within a process and would favour whichever configuration runs second.

    python eval/bench_e2e.py --frames 200 --label gpavggt
    python eval/bench_e2e.py --frames 200 --label vanilla --vanilla

Prints an `E2E ...` line and appends one row to results/e2e_latency.csv.
Model flags are the same as eval_co3d.py.
"""

import argparse
import csv
import gc
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import eval_co3d as E  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "e2e_latency.csv"
FIELDS = ["label", "frames", "e2e_ms", "agg_ms", "e2e_peak_gib", "iters"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames", type=int, default=32)
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--img_size", type=int, default=518)
    p.add_argument("--label", required=True)
    p.add_argument("--time_agg", action="store_true",
                   help="also time the aggregator alone")
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--model_path", type=str,
                   default=os.path.join(E.ROOT_DIR, "ckpt", "model_tracker_fixed_e20.pt"))
    E.add_model_args(p)
    return p.parse_args()


def timed(fn, warmup, iters):
    """(mean ms, peak GiB of the measured window)."""
    out = None
    for _ in range(warmup):
        del out
        out = fn()
    torch.cuda.synchronize()
    del out
    out = None
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(iters):
        del out
        out = fn()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1000
    peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    del out
    gc.collect()
    torch.cuda.empty_cache()
    return ms, peak


def main():
    args = parse_args()
    device, dtype = "cuda", torch.bfloat16
    torch.manual_seed(0)

    model = E.build_model(args, device, dtype)

    s, size = args.frames, args.img_size
    images = torch.rand(1, s, 3, size, size, device=device, dtype=dtype)
    model.update_patch_dimensions(size // 14, size // 14)

    def aggregator():
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
            out, _ = model.aggregator(images)
        return out

    def e2e():
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
            toks, psi = model.aggregator(images)
            pose = model.camera_head(toks)
            depth, conf = model.depth_head(toks, images=images, patch_start_idx=psi)
        return pose, depth, conf

    agg_ms = timed(aggregator, args.warmup, args.iters)[0] if args.time_agg else None
    e2e_ms, e2e_peak = timed(e2e, args.warmup, args.iters)

    row = {"label": args.label, "frames": s, "e2e_ms": round(e2e_ms, 1),
           "agg_ms": round(agg_ms, 1) if agg_ms is not None else "-",
           "e2e_peak_gib": round(e2e_peak, 2), "iters": args.iters}
    print(f"E2E label={args.label} frames={s} e2e_ms={e2e_ms:.1f} "
          + (f"agg_ms={agg_ms:.1f} " if agg_ms is not None else "")
          + f"e2e_peak_gib={e2e_peak:.2f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    new = not out_path.exists()
    with out_path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


if __name__ == "__main__":
    main()
