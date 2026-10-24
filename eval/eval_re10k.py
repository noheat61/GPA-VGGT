"""RealEstate10K relative-pose evaluation (AUC@N).

Same metric and scoring path as eval_co3d.py -- only the data source differs, so
the two tables are read the same way. Frames come from `re10k_fetch.py`, which
writes one directory per sequence:

    <root>/<seq>/frames/<timestamp_us>.jpg
    <root>/<seq>/cameras.txt      # URL, then one RealEstate10K pose line per frame

A pose line is `timestamp fx fy cx cy 0 0` followed by the 12 entries of the
world-to-camera 3x4 matrix, row-major. Intrinsics are normalised by image size
and unused here: relative pose error needs extrinsics only.

    python eval/eval_re10k.py --re10k_dir <root> --num_frames 10 --deterministic

Model flags (including --vanilla) are the same as eval_co3d.py.
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import eval_co3d as C  # noqa: E402
M = C.M


def parse_args():
    p = argparse.ArgumentParser(description="RealEstate10K relative-pose eval")
    p.add_argument("--re10k_dir", type=str,
                   default=os.path.join(C.ROOT_DIR, "dataset", "re10k", "frames"))
    p.add_argument("--model_path", type=str,
                   default=os.path.join(C.ROOT_DIR, "ckpt", "model_tracker_fixed_e20.pt"))
    p.add_argument("--num_frames", type=int, default=10)
    p.add_argument("--num_seq", type=int, default=0, help="0 = every fetched sequence")
    p.add_argument("--per_seq_csv", default="", help="write per-sequence AUC/RRA/RTA")
    p.add_argument("--frame_sample", default="uniform", choices=["uniform", "random"],
                   help="uniform spreads the frames over the clip (our default); "
                        "random reproduces the published 10-random-frames protocol")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deterministic", action="store_true", default=False)
    p.add_argument("--pool_pairs", action="store_true", default=False,
                   help="score every pair of every sequence in one pool, the way "
                        "the pi3 evaluation branch aggregates (relpose/eval_angle.py "
                        "extends one rError/tError list across sequences). Our "
                        "default macro-averages sequences, which is what the CO3D "
                        "table does; a long sequence carries more weight under "
                        "pooling, so the two are not interchangeable")

    C.add_model_args(p)
    return p.parse_args()


def load_sequence(seq_dir, num_frames, rng=None):
    """-> (image paths, [4x4 world-to-camera]) or (None, None) if unusable."""
    cam = seq_dir / "cameras.txt"
    if not cam.exists():
        return None, None
    lines = [l for l in cam.read_text().splitlines()[1:] if l.strip()]

    names, extri = [], []
    for line in lines:
        f = line.split()
        img = seq_dir / "frames" / f"{f[0]}.jpg"
        if not img.exists():          # a frame ffmpeg could not pull
            continue
        names.append(str(img))
        extri.append(np.asarray(f[7:19], dtype=np.float64).reshape(3, 4))

    if len(names) < num_frames:
        return None, None
    # cameras.txt already holds the sampled frames in order; if a fetch produced
    # extras, keep the same uniform spread the fetcher used -- unless a generator
    # is passed, which switches to the published protocol (VGGT Tab. 1 and every
    # paper that follows it: "we randomly select 10 images"). Random draws cluster
    # in time far more often than a uniform spread does, so the two are not
    # interchangeable: uniform is the harder, wider-baseline setting.
    if len(names) > num_frames:
        if rng is not None:
            idx = np.sort(rng.choice(len(names), num_frames, replace=False))
        else:
            idx = np.linspace(0, len(names) - 1, num_frames).round().astype(int)
        names = [names[i] for i in idx]
        extri = [extri[i] for i in idx]
    return names, np.stack(extri)


def main():
    args = parse_args()
    if args.deterministic:
        C.enable_determinism()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16
    torch.set_float32_matmul_precision("highest")

    from pathlib import Path
    root = Path(args.re10k_dir)
    seqs = sorted(d for d in root.iterdir() if d.is_dir() and (d / "cameras.txt").exists())
    np.random.seed(args.seed)
    if args.num_seq:
        seqs = seqs[:args.num_seq]
    # One generator for the whole sweep, consumed in sorted sequence order, so a
    # rerun with the same --seed draws the same frames for every method.
    frame_rng = (np.random.default_rng(args.seed)
                 if args.frame_sample == "random" else None)
    print(f"device: {device} | sequences={len(seqs)} num_frames={args.num_frames} "
          f"frame_sample={args.frame_sample}")

    model = C.build_model(args, device, dtype)

    THRESHOLDS = (30, 15, 10, 5, 3)
    per_seq, used, skipped = [], 0, 0
    pool_r, pool_t = [], []
    for i, seq_dir in enumerate(seqs, 1):
        names, gt_extri = load_sequence(seq_dir, args.num_frames, frame_rng)
        if names is None:
            skipped += 1
            continue

        pred = C.predict_extrinsics(model, names, device, dtype)
        with torch.cuda.amp.autocast(dtype=torch.float64):
            pred_t = torch.as_tensor(pred, device=device, dtype=torch.float64)
            gt_t = torch.from_numpy(gt_extri).to(device)
            row = torch.tensor([0, 0, 0, 1], device=device).expand(pred_t.size(0), 1, 4)
            rel_r, rel_t = M.se3_to_relative_pose_error(
                torch.cat((pred_t, row), dim=1),
                torch.cat((gt_t, row), dim=1),
                args.num_frames,
            )
        # AUC per sequence, then averaged over sequences -- the aggregation
        # eval_co3d.py uses. Pooling every pair first would weight long
        # sequences more and is not comparable to the CO3D table.
        rr, tt = rel_r.cpu().numpy(), rel_t.cpu().numpy()
        pool_r.append(rr)
        pool_t.append(tt)
        row_out = {th: M.calculate_auc_np(rr, tt, th)[0] for th in THRESHOLDS}
        # RRA@30 / RTA@30 as published tables report them: the fraction of pairs
        # under 30 deg, rotation and translation separately. AUC alone cannot say
        # which of the two a gap lives in, and on this benchmark that is the whole
        # question -- rotation is nearly saturated in every published row.
        row_out["rra30"] = float((rr < 30).mean())
        row_out["rta30"] = float((tt < 30).mean())
        row_out["seq"] = seq_dir.name
        per_seq.append(row_out)
        used += 1
        if i % 50 == 0:
            print(f"  {i}/{len(seqs)} used={used} skipped={skipped}", flush=True)

    if not used:
        raise SystemExit("no usable sequences")

    if args.pool_pairs:
        rr, tt = np.concatenate(pool_r), np.concatenate(pool_t)
        aucs = {th: float(M.calculate_auc_np(rr, tt, th)[0]) for th in THRESHOLDS}
        rra, rta = float((rr < 30).mean()), float((tt < 30).mean())
    else:
        aucs = {th: float(np.mean([s[th] for s in per_seq])) for th in THRESHOLDS}
        rra = float(np.mean([s["rra30"] for s in per_seq]))
        rta = float(np.mean([s["rta30"] for s in per_seq]))
    print("-" * 50)
    print(f"sequences used={used} skipped={skipped} "
          f"aggregation={'pooled pairs' if args.pool_pairs else 'macro over sequences'} "
          f"pairs={sum(len(x) for x in pool_r)}")
    print("Mean AUC: " + ", ".join(f"{aucs[th]:.4f} (AUC@{th})" for th in THRESHOLDS))
    print(f"RRA@30={rra * 100:.2f} RTA@30={rta * 100:.2f}")
    print("FINAL_MEAN_AUC " + " ".join(f"AUC{th}={aucs[th]:.4f}" for th in (30, 10, 5, 3))
          + f" RRA30={rra:.4f} RTA30={rta:.4f}")

    if args.per_seq_csv:
        import csv
        with open(args.per_seq_csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["seq"] + [f"auc{th}" for th in THRESHOLDS] + ["rra30", "rta30"])
            for s in per_seq:
                w.writerow([s["seq"]] + [f"{s[th]:.6f}" for th in THRESHOLDS]
                           + [f"{s['rra30']:.6f}", f"{s['rta30']:.6f}"])
        print(f"wrote {args.per_seq_csv}")


if __name__ == "__main__":
    main()
