"""Multi-view point-cloud reconstruction: Accuracy, Completeness, Normal Consistency.

    # Pi3-eval mv_recon protocol -- the published seq-id map, whole frame
    # (needs Pi3-eval cloned next to this repository for the seq-id maps)
    python eval/eval_mv_recon.py --dataset nrgbd --protocol pi3 --kf 100 --deterministic

    # LiteVGGT Tab. 2 protocol -- whole video at stride 3, centre 224 crop
    python eval/eval_mv_recon.py --dataset nrgbd --protocol litevggt --kf 3 --deterministic

Model flags (including --vanilla) are the same as eval_co3d.py.

Both are the Spann3R metric -- nearest-neighbour distance in each direction plus
the absolute dot product of the matched normals -- after the same two-stage
alignment: a similarity fit (umeyama, scale included) on every valid point,
then point-to-point ICP at a 0.1 m threshold. `eval/mv_recon_data.py` documents
where the two protocols part company; the short version is stride and crop, and
they are far enough apart that their tables must never be merged.

The prediction is the depth head unprojected with the predicted pose, not the
point head, because that is what LiteVGGT's `eval/eval_7andN.py` scores and the
two heads do not agree closely enough for the choice to be cosmetic.

Nothing here filters by predicted confidence. Both reference implementations
carry the confidence into the loop and then never threshold it -- LiteVGGT's
`mask = mask & (conf > 1.8)` is commented out in its own source -- so the mask is
the ground truth's validity and nothing else. `--conf_thresh` exists to measure
what that choice is worth, and defaults to 0.

Determinism note: the 1e6-point cap under `--protocol litevggt` subsamples with
`np.random`, seeded per sequence from `--seed`, so two methods see the same
points. Without the cap (Pi3 protocol) nothing here is random.
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import eval_co3d as C  # noqa: E402
import mv_recon_data as MD  # noqa: E402

DEFAULT_ROOT = {"nrgbd": os.path.join(C.ROOT_DIR, "dataset", "nrgbd_full"),
                "sevenscenes": os.path.join(C.ROOT_DIR, "dataset",
                                            "seven_scenes_full")}
# The published maps, so a Pi3-protocol rerun cannot drift with a directory listing.
PI3_MAPS = os.path.join(C.ROOT_DIR, os.pardir, "Pi3-eval", "datasets",
                        "seq-id-maps")
PI3_MAP_NAME = {("nrgbd", 500): "NRGBD_mv-recon_seq-id-map-kf500.json",
                ("nrgbd", 100): "NRGBD_mv-recon_seq-id-map-kf100.json",
                ("sevenscenes", 200): "7scenes_mv-recon_seq-id-map-kf200.json",
                ("sevenscenes", 40): "7scenes_mv-recon_seq-id-map-kf40.json"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=sorted(DEFAULT_ROOT))
    p.add_argument("--protocol", default="pi3", choices=("pi3", "litevggt"))
    p.add_argument("--kf", type=int, default=0,
                   help="stride. pi3: 500/100 (nrgbd) or 200/40 (7scenes), which "
                        "selects the published map. litevggt: 3, as its README runs it")
    p.add_argument("--data_dir", default=None)
    p.add_argument("--seq_id_map", default=None, help="override the published map")
    p.add_argument("--num_seq", type=int, default=0)
    p.add_argument("--max_frames", type=int, default=0,
                   help="cap frames per sequence (0 = the protocol's own count). "
                        "A cap makes the row incomparable to a published one; it "
                        "is here so an OOM can be reported as a number rather "
                        "than a missing row")
    p.add_argument("--crop", type=int, default=0,
                   help="centre crop before scoring; litevggt implies 224")
    p.add_argument("--point_cap", type=int, default=-1,
                   help="-1 = the protocol's default (1e6 for litevggt, none for pi3)")
    p.add_argument("--conf_thresh", type=float, default=0.0,
                   help="drop pixels below this depth confidence. Neither "
                        "reference implementation does; 0 keeps all")
    p.add_argument("--icp_thresh", type=float, default=0.1)
    p.add_argument("--per_seq_csv", default="")
    p.add_argument("--save_ply", default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deterministic", action="store_true", default=False)
    p.add_argument("--model_path", type=str,
                   default=os.path.join(C.ROOT_DIR, "ckpt",
                                        "model_tracker_fixed_e20.pt"))

    C.add_model_args(p)
    return p.parse_args()


# --------------------------------------------------------------------------
# metrics -- transcribed from Pi3-eval/mv_recon/utils.py, which is Spann3R's
# --------------------------------------------------------------------------

def umeyama(X, Y):
    """Similarity transform taking X (3,N) onto Y (3,N): Y ~ c * R @ X + t."""
    mu_x, mu_y = X.mean(axis=1)[:, None], Y.mean(axis=1)[:, None]
    var_x = np.square(X - mu_x).sum(axis=0).mean()
    cov_xy = ((Y - mu_y) @ (X - mu_x).T) / X.shape[1]
    U, D, VH = np.linalg.svd(cov_xy)
    S = np.eye(X.shape[0])
    # Guards against the reflection SVD is free to return when the covariance is
    # near-degenerate; without it a mirrored fit can score better than the truth.
    if np.linalg.det(U) * np.linalg.det(VH) < 0:
        S[-1, -1] = -1
    c = np.trace(np.diag(D) @ S) / var_x
    R = U @ S @ VH
    t = mu_y - c * R @ mu_x
    return c, R, t


def _nn(tree, query):
    d, i = tree.query(query)
    return d, i


def accuracy(gt_points, rec_points, gt_normals, rec_normals):
    """Prediction -> ground truth: how far a predicted point is from the surface."""
    from scipy.spatial import cKDTree as KDTree
    d, idx = KDTree(np.asarray(gt_points)).query(np.asarray(rec_points))
    dot = np.abs(np.sum(np.asarray(gt_normals)[idx] * np.asarray(rec_normals),
                        axis=-1))
    return d.mean(), np.median(d), dot.mean(), np.median(dot)


def completion(gt_points, rec_points, gt_normals, rec_normals):
    """Ground truth -> prediction: how much of the surface was reconstructed."""
    from scipy.spatial import cKDTree as KDTree
    d, idx = KDTree(np.asarray(rec_points)).query(np.asarray(gt_points))
    dot = np.abs(np.sum(np.asarray(gt_normals) * np.asarray(rec_normals)[idx],
                        axis=-1))
    return d.mean(), np.median(d), dot.mean(), np.median(dot)


def _centre_crop(arr, size):
    """Centre `size` x `size` window of an (..., H, W, C) or (..., H, W) array."""
    h, w = arr.shape[1:3]
    t, l = h // 2 - size // 2, w // 2 - size // 2
    if t < 0 or l < 0:
        raise SystemExit(f"--crop {size} does not fit in {h}x{w}")
    return arr[:, t:t + size, l:l + size]


def main():
    args = parse_args()
    if args.deterministic:
        C.enable_determinism()

    import open3d as o3d

    root = args.data_dir or DEFAULT_ROOT[args.dataset]
    if not os.path.isdir(root):
        raise SystemExit(f"{args.dataset}: {root} does not exist")

    kf = args.kf or (3 if args.protocol == "litevggt"
                     else (500 if args.dataset == "nrgbd" else 200))
    crop = args.crop or (224 if args.protocol == "litevggt" else 0)
    cap = args.point_cap
    if cap < 0:
        cap = 999999 if args.protocol == "litevggt" else 0

    seq_id_map = None
    if args.protocol == "pi3":
        path = args.seq_id_map or os.path.join(
            PI3_MAPS, PI3_MAP_NAME.get((args.dataset, kf), ""))
        if not os.path.exists(path):
            raise SystemExit(
                f"--protocol pi3 needs the published seq-id map for "
                f"{args.dataset} kf{kf}; looked at {path}")
        import json
        with open(path) as fh:
            seq_id_map = json.load(fh)
        print(f"seq-id map: {path}")

    seqs = MD.sequences(args.dataset, root)
    if args.protocol == "pi3":
        seqs = [s for s in seqs if s in seq_id_map]
    if args.num_seq:
        seqs = seqs[:args.num_seq]
    if not seqs:
        raise SystemExit("no sequences")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16
    torch.set_float32_matmul_precision("highest")
    print(f"device: {device} | dataset={args.dataset} protocol={args.protocol} "
          f"kf={kf} crop={crop or 'none'} cap={cap or 'none'} "
          f"sequences={len(seqs)}", flush=True)

    model = C.build_model(args, device, dtype)
    if getattr(model, "depth_head", None) is None:
        raise SystemExit("this model was built without a depth head")

    pose_enc_to = C.pose_encoding_to_extri_intri

    rows, oom, skipped = [], [], []
    for si, seq in enumerate(seqs, 1):
        ids = MD.frame_ids(args.dataset, root, seq, args.protocol, kf, seq_id_map)
        if not ids:
            skipped.append(seq)
            print(f"  [{si}/{len(seqs)}] {seq}: no frames", flush=True)
            continue
        if args.max_frames and len(ids) > args.max_frames:
            ids = ids[:args.max_frames]
        # A scene that will not fit is a result, not a crash: at stride 3 an
        # NRGBD sequence reaches 552 frames, and where a method stops fitting is
        # part of what this table reports. The sequence is dropped and named, and
        # the row's `sequences=` count then says the mean is over a smaller set --
        # which is exactly what makes such a row incomparable to a complete one.
        try:
            row = _one_sequence(args, model, pose_enc_to, root, seq, ids, crop,
                                cap, si, device, dtype, o3d)
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            oom.append(seq)
            print(f"  [{si}/{len(seqs)}] {seq}: OOM at {len(ids)} frames -- "
                  f"{str(exc).splitlines()[0]}", flush=True)
            continue
        if row is None:
            skipped.append(seq)
            print(f"  [{si}/{len(seqs)}] {seq}: too little valid ground truth",
                  flush=True)
            continue
        rows.append(row)
        print(f"  [{si}/{len(seqs)}] {seq}: frames={row['frames']} "
              f"pts={row['points']} Acc={row['acc']:.4f} Comp={row['comp']:.4f} "
              f"NC={row['nc']:.4f} {row['infer_ms'] / 1000:.1f}s", flush=True)

    if not rows:
        raise SystemExit("no sequence produced a usable reconstruction")

    keys = ["acc", "acc_med", "comp", "comp_med", "nc", "nc_med",
            "nc1", "nc1_med", "nc2", "nc2_med"]
    mean = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    secs = float(np.mean([r["infer_ms"] for r in rows])) / 1000.0
    print("-" * 70)
    print(f"sequences={len(rows)} frames={sum(r['frames'] for r in rows)}")
    if oom:
        print(f"OOM: {len(oom)} sequence(s) -- {', '.join(oom)}")
    if skipped:
        print(f"skipped: {len(skipped)} sequence(s) -- {', '.join(skipped)}")
    print(f"FINAL_MV_RECON acc={mean['acc']:.4f} acc_med={mean['acc_med']:.4f} "
          f"comp={mean['comp']:.4f} comp_med={mean['comp_med']:.4f} "
          f"nc={mean['nc']:.4f} nc1={mean['nc1']:.4f} nc2={mean['nc2']:.4f} "
          f"secs={secs:.2f}")

    if args.per_seq_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.per_seq_csv)),
                    exist_ok=True)
        with open(args.per_seq_csv, "w", newline="") as fh:
            w = csv.DictWriter(
                fh, fieldnames=["seq", "frames", "points", "infer_ms"] + keys)
            w.writeheader()
            for r in rows:
                w.writerow({**r, **{k: f"{r[k]:.6f}" for k in keys}})
        print(f"wrote {args.per_seq_csv}")


def _one_sequence(args, model, pose_enc_to, root, seq, ids, crop, cap, si,
                  device, dtype, o3d):
    """Score one sequence, or None if it carries too little valid ground truth."""
    from vggt.utils.geometry import unproject_depth_map_to_point_map

    data = MD.load(args.dataset, root, seq, ids)
    gt_pts = data["pointclouds"]                       # (N, H, W, 3)
    valid = data["valid_mask"]                         # (N, H, W)

    # The dataset's own tensor, not `load_and_preprocess_images`: both reference
    # implementations feed the loader's images straight to the model, and the two
    # resamplers disagree. Pi3's `resize_image` picks LANCZOS whenever it is
    # downscaling (640 -> 518 is), while VGGT's loader is BICUBIC unconditionally.
    # On a stable scene that is worth ~1% of Acc; on grey_white_room, where the
    # reconstruction is already marginal, it moved Acc from 0.149 to 0.159 -- and
    # 0.149 is what reproduces the earlier run. Using the loader's images also
    # makes prediction and ground truth pixel-aligned by construction.
    images = data["images"].to(device)
    if tuple(images.shape[-2:]) != tuple(gt_pts.shape[1:3]):
        raise SystemExit(f"{seq}: model grid {tuple(images.shape[-2:])} != "
                         f"ground truth {gt_pts.shape[1:3]}")
    if device == "cuda":
        images = images.to(dtype)
    _, _, H, W = images.shape
    if hasattr(model, "update_patch_dimensions"):
        model.update_patch_dimensions(W // 14, H // 14)

    # Wall clock of the forward alone -- the column LiteVGGT Tab. 2 calls Time.
    # Only meaningful when this queue has the GPU to itself; the loader, the
    # KD-trees and ICP all sit outside it.
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
        pred = model(images)
    torch.cuda.synchronize()
    infer_ms = (time.perf_counter() - t0) * 1000.0

    extri, intri = pose_enc_to(pred["pose_enc"], (H, W))
    pts = unproject_depth_map_to_point_map(
        pred["depth"].squeeze(0).float().cpu().numpy(),
        extri.squeeze(0).float().cpu().numpy(),
        intri.squeeze(0).float().cpu().numpy())            # (N, H, W, 3)
    if args.conf_thresh > 0:
        valid = valid & (pred["depth_conf"].squeeze(0).float().cpu().numpy()
                         > args.conf_thresh)
    del pred, images
    torch.cuda.empty_cache()

    colors = data["images"].permute(0, 2, 3, 1).numpy()
    if crop:
        pts, gt_pts = _centre_crop(pts, crop), _centre_crop(gt_pts, crop)
        valid, colors = _centre_crop(valid, crop), _centre_crop(colors, crop)
    if valid.sum() < 100:
        return None

    # Coarse similarity fit on every valid point, then keep only those.
    c, R, t = umeyama(pts[valid].T, gt_pts[valid].T)
    pts = c * np.einsum("nhwj, ij -> nhwi", pts, R) + t.T
    pts = pts[valid].reshape(-1, 3)
    gt = gt_pts[valid].reshape(-1, 3)
    colors = colors[valid].reshape(-1, 3)

    if cap and pts.shape[0] > cap:
        # One generator per sequence, seeded the same way for every method, so
        # two rows are subsampled identically. Prediction and ground truth are
        # drawn independently, as LiteVGGT does it -- a shared draw would pair up
        # points the metric is supposed to match by distance.
        rng = np.random.default_rng(args.seed + si)
        keep = rng.choice(pts.shape[0], cap, replace=False)
        pts, colors = pts[keep], colors[keep]
    if cap and gt.shape[0] > cap:
        rng = np.random.default_rng(args.seed + si + 100000)
        gt = gt[rng.choice(gt.shape[0], cap, replace=False)]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(gt)

    reg = o3d.pipelines.registration.registration_icp(
        pcd, pcd_gt, args.icp_thresh, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    pcd = pcd.transform(reg.transformation)
    pcd.estimate_normals()
    pcd_gt.estimate_normals()
    pn, gn = np.asarray(pcd.normals), np.asarray(pcd_gt.normals)

    acc, acc_med, nc1, nc1_med = accuracy(pcd_gt.points, pcd.points, gn, pn)
    comp, comp_med, nc2, nc2_med = completion(pcd_gt.points, pcd.points, gn, pn)
    if args.save_ply:
        os.makedirs(args.save_ply, exist_ok=True)
        o3d.io.write_point_cloud(
            os.path.join(args.save_ply, f"{seq.replace('/', '-')}-pred.ply"), pcd)
    return {"seq": seq, "frames": len(ids), "points": len(pts),
            "infer_ms": infer_ms,
            "acc": acc, "acc_med": acc_med, "comp": comp, "comp_med": comp_med,
            "nc": (nc1 + nc2) / 2, "nc_med": (nc1_med + nc2_med) / 2,
            "nc1": nc1, "nc1_med": nc1_med, "nc2": nc2, "nc2_med": nc2_med}


if __name__ == "__main__":
    main()
