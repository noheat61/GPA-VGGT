"""CO3D relative-pose evaluation (AUC@N) for VGGT and GPA-VGGT.

    # GPA-VGGT (the model defaults are the paper's configuration)
    python eval/eval_co3d.py --fast_eval --num_frames 10 --seed 0 --deterministic

    # vanilla VGGT
    python eval/eval_co3d.py --fast_eval --num_frames 10 --seed 0 --deterministic --vanilla

Point --co3d_dir / --co3d_anno_dir at the CO3D test split, or symlink it to
`dataset/co3d` inside the repo (the default).
"""

import argparse
import csv
import gzip
import json
import os
import random
import sys

import numpy as np
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pose_metrics as M  # noqa: E402
from vggt.models.vggt import VGGT  # noqa: E402
from vggt.utils.load_fn import load_and_preprocess_images  # noqa: E402
from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # noqa: E402

SEEN_CATEGORIES = [
    "apple", "backpack", "banana", "baseballbat", "baseballglove",
    "bench", "bicycle", "bottle", "bowl", "broccoli",
    "cake", "car", "carrot", "cellphone", "chair",
    "cup", "donut", "hairdryer", "handbag", "hydrant",
    "keyboard", "laptop", "microwave", "motorcycle", "mouse",
    "orange", "parkingmeter", "pizza", "plant", "stopsign",
    "teddybear", "toaster", "toilet", "toybus", "toyplane",
    "toytrain", "toytruck", "tv", "umbrella", "vase", "wineglass",
]


def enable_determinism():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="CO3D relative-pose eval")
    parser.add_argument("--model_path", type=str,
                        default=os.path.join(ROOT_DIR, "ckpt", "model_tracker_fixed_e20.pt"),
                        help="Checkpoint path")
    parser.add_argument("--co3d_dir", type=str,
                        default=os.path.join(ROOT_DIR, "dataset", "co3d", "DATA"),
                        help="CO3D images root")
    parser.add_argument("--co3d_anno_dir", type=str,
                        default=os.path.join(ROOT_DIR, "dataset", "co3d", "ANNO"),
                        help="CO3D annotation (*_test.jgz) root")
    parser.add_argument("--fast_eval", action="store_true", default=False,
                        help="Only evaluate 10 random sequences per category")
    parser.add_argument("--min_num_images", type=int, default=64,
                        help="Skip sequences with fewer images than this")
    parser.add_argument("--num_frames", type=int, default=10,
                        help="Frames sampled per sequence")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true", default=False,
                        help="Force deterministic kernels. Token merging uses "
                             "scatter_reduce(mean), whose CUDA atomics accumulate in a "
                             "run-dependent order; without this, repeated runs of the same "
                             "merging config differ by up to ~0.35 pt AUC@3, which is larger "
                             "than most effects being measured. Needs "
                             "CUBLAS_WORKSPACE_CONFIG=:4096:8 in the environment.")
    parser.add_argument("--categories", nargs="+", default=None,
                        help="Explicit category subset to evaluate. Note that this "
                             "changes which sequences every later category samples, "
                             "so runs with different subsets are not comparable.")
    parser.add_argument("--seqs_per_category", type=int, default=0,
                        help="keep only the first N of each category's "
                             "fast_eval draw (0 = all). See the note at "
                             "the truncation site.")
    parser.add_argument("--per_sequence_csv", default=None,
                        help="Optional CSV with one row per evaluated sequence, including "
                             "AUC thresholds and the median GT pairwise rotation baseline.")

    add_model_args(parser)
    return parser.parse_args()


def add_model_args(parser):
    """Model flags shared by every evaluation script. The defaults are the
    GPA-VGGT configuration used in the paper."""
    parser.add_argument("--vanilla", action="store_true", default=False,
                        help="Run vanilla VGGT (no frame attention, no merging).")
    parser.add_argument("--early_frame_split", type=int, default=9,
                        help="Global blocks with index < N run frame attention plus a "
                             "shared cross-frame K/V set.")
    parser.add_argument("--broadcast_pool", type=int, default=8,
                        help="Seed stride of the pooled patch summaries in the shared "
                             "set (0 = special tokens only, -1 = no shared set).")
    parser.add_argument("--merging", type=int, default=9,
                        help="First global block with K/V merging (-1 disables it).")
    parser.add_argument("--merge_end", type=int, default=23,
                        help="Last global block with K/V merging (inclusive).")
    parser.add_argument("--merge_ratio", type=float, default=0.98,
                        help="Fraction of tokens offered for merging.")
    parser.add_argument("--core_blocks", type=int, nargs="*", default=[13, 14, 15, 16],
                        help="Global blocks merged at --core_merge_ratio instead.")
    parser.add_argument("--core_merge_ratio", type=float, default=0.2,
                        help="Merge ratio of the core blocks (0 = not merged).")
    parser.add_argument("--merge_metric_dim", type=int, default=128,
                        help="Random-projection dimension of the matching metric "
                             "(0 = full dimension).")


def build_model(args, device, dtype):
    if args.vanilla:
        cfg = dict(merging=None, early_frame_split=0)
    else:
        cfg = dict(merging=None if args.merging < 0 else args.merging,
                   merge_end=args.merge_end, merge_ratio=args.merge_ratio,
                   early_frame_split=args.early_frame_split,
                   broadcast_pool=args.broadcast_pool,
                   core_blocks=tuple(args.core_blocks or ()),
                   core_merge_ratio=args.core_merge_ratio,
                   merge_metric_dim=args.merge_metric_dim)
    print(f"Initializing {'vanilla VGGT' if args.vanilla else 'GPA-VGGT'} {cfg}")
    # eval_mv_recon.py is the one caller that needs the point head.
    model = VGGT(enable_track=False, enable_point=getattr(args, "need_point", False), **cfg)

    print(f"USING {args.model_path}")
    state_dict = torch.load(args.model_path, map_location="cpu")
    model_keys = set(model.state_dict().keys())
    filtered = {k: v for k, v in state_dict.items() if k in model_keys}
    dropped = len(state_dict) - len(filtered)
    if dropped:
        print(f"Skipping {dropped} checkpoint keys for disabled heads.")
    model.load_state_dict(filtered, strict=False)

    model.eval().to(device)
    if device == "cuda":
        model = model.to(dtype)
    return model


def predict_extrinsics(model, image_names, device, dtype):
    """Run the model on a list of image paths -> (N, 3, 4) OpenCV world-to-camera."""
    images = load_and_preprocess_images(image_names).to(device)
    if device == "cuda":
        images = images.to(dtype)

    # The attention/merging layers need the current patch grid dimensions.
    _, _, H, W = images.shape
    model.update_patch_dimensions(W // 14, H // 14)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)
    with torch.cuda.amp.autocast(dtype=torch.float64):
        extrinsic, _ = pose_encoding_to_extri_intri(
            predictions["pose_enc"].float(), images.shape[-2:]
        )
    return extrinsic[0].cpu().numpy()


def process_sequence(model, seq_name, seq_data, category, args, device, dtype):
    """Return pair errors plus the GT rotation baseline, or three Nones."""
    if len(seq_data) < args.min_num_images:
        return None, None, None

    metadata = []
    for data in seq_data:
        if data["T"][0] + data["T"][1] + data["T"][2] > 1e5:
            return None, None, None
        extri_opencv = M.convert_pt3d_RT_to_opencv(data["R"], data["T"])
        metadata.append({"filepath": data["filepath"], "extri": extri_opencv})

    ids = np.random.choice(len(metadata), args.num_frames, replace=False)
    print("Image ids", ids)

    image_names = [os.path.join(args.co3d_dir, metadata[i]["filepath"]) for i in ids]
    gt_extri = np.stack([np.array(metadata[i]["extri"]) for i in ids], axis=0)

    pred_extrinsic = predict_extrinsics(model, image_names, device, dtype)

    with torch.cuda.amp.autocast(dtype=torch.float64):
        pred_extrinsic = torch.as_tensor(pred_extrinsic, device=device, dtype=torch.float64)
        gt_extrinsic = torch.from_numpy(gt_extri).to(device)
        add_row = torch.tensor([0, 0, 0, 1], device=device).expand(pred_extrinsic.size(0), 1, 4)

        pred_se3 = torch.cat((pred_extrinsic, add_row), dim=1)
        gt_se3 = torch.cat((gt_extrinsic, add_row), dim=1)

        rel_rangle_deg, rel_tangle_deg = M.se3_to_relative_pose_error(
            pred_se3, gt_se3, args.num_frames
        )
        pair_i, pair_j = M.build_pair_index(args.num_frames)
        relative_gt = gt_se3[pair_i].bmm(M.closed_form_inverse_se3(gt_se3[pair_j]))
        identity = torch.eye(3, device=device, dtype=torch.float64).expand(len(pair_i), 3, 3)
        gt_rotation_deg = M.rotation_angle(relative_gt[:, :3, :3], identity)

        Racc_5 = (rel_rangle_deg < 5).float().mean().item()
        Tacc_5 = (rel_tangle_deg < 5).float().mean().item()
        print(f"{category} sequence {seq_name} R_ACC@5: {Racc_5:.4f}")
        print(f"{category} sequence {seq_name} T_ACC@5: {Tacc_5:.4f}")

        return (rel_rangle_deg.cpu().numpy(), rel_tangle_deg.cpu().numpy(),
                gt_rotation_deg.cpu().numpy())


def main():
    args = parse_args()
    if args.deterministic:
        enable_determinism()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False

    print(f"device: {device} | min_num_images={args.min_num_images} "
          f"num_frames={args.num_frames}")
    model = build_model(args, device, dtype)

    set_random_seeds(args.seed)

    if args.categories:
        categories = args.categories
    else:
        categories = SEEN_CATEGORIES
    per_category_results = {}
    per_sequence_results = []

    for category in categories:
        print(f"Loading annotation for {category} test set")
        annotation_file = os.path.join(args.co3d_anno_dir, f"{category}_test.jgz")
        try:
            with gzip.open(annotation_file, "r") as fin:
                annotation = json.loads(fin.read())
        except FileNotFoundError:
            print(f"Annotation file not found for {category}, skipping")
            continue

        seq_names = sorted(list(annotation.keys()))
        if args.fast_eval and len(seq_names) >= 10:
            seq_names = random.sample(seq_names, 10)
        seq_names = sorted(seq_names)
        # Truncate after random.sample so every category's draw stays the same.
        if args.seqs_per_category:
            seq_names = seq_names[:args.seqs_per_category]
        print("Testing Sequences:")
        print(seq_names)

        rError, tError = [], []
        for seq_name in seq_names:
            print("-" * 50)
            print(f"Processing {seq_name} for {category} test set")
            if not os.path.exists(os.path.join(args.co3d_dir, category, seq_name)):
                print(f"Skipping {seq_name} (not found)")
                continue
            seq_rError, seq_tError, seq_gt_rotation = process_sequence(
                model, seq_name, annotation[seq_name], category, args, device, dtype
            )
            print("-" * 50)
            if seq_rError is not None:
                rError.extend(seq_rError)
                tError.extend(seq_tError)
                per_sequence_results.append({
                    "category": category,
                    "sequence": seq_name,
                    "num_frames": args.num_frames,
                    "num_pairs": len(seq_rError),
                    "gt_rotation_median_deg": float(np.median(seq_gt_rotation)),
                    "gt_rotation_mean_deg": float(np.mean(seq_gt_rotation)),
                    "AUC30": M.calculate_auc_np(seq_rError, seq_tError, 30)[0],
                    "AUC15": M.calculate_auc_np(seq_rError, seq_tError, 15)[0],
                    "AUC10": M.calculate_auc_np(seq_rError, seq_tError, 10)[0],
                    "AUC5": M.calculate_auc_np(seq_rError, seq_tError, 5)[0],
                    "AUC3": M.calculate_auc_np(seq_rError, seq_tError, 3)[0],
                    "RRA30": float((np.asarray(seq_rError) < 30).mean()),
                    "RTA30": float((np.asarray(seq_tError) < 30).mean()),
                })

        if not rError:
            print(f"No valid sequences found for {category}, skipping")
            continue

        rError = np.array(rError)
        tError = np.array(tError)
        res = {
            "Auc_30": M.calculate_auc_np(rError, tError, 30)[0],
            "Auc_15": M.calculate_auc_np(rError, tError, 15)[0],
            "Auc_10": M.calculate_auc_np(rError, tError, 10)[0],
            "Auc_5": M.calculate_auc_np(rError, tError, 5)[0],
            "Auc_3": M.calculate_auc_np(rError, tError, 3)[0],
            # Averaged over categories exactly like the AUCs below, so every
            # number on the summary line shares one aggregation.
            "Rra_30": float((rError < 30).mean()),
            "Rta_30": float((tError < 30).mean()),
        }
        per_category_results[category] = res

        print("=" * 80)
        print(f"AUC of {category} test set: {res['Auc_30']:.4f} (AUC@30), "
              f"{res['Auc_15']:.4f} (AUC@15), {res['Auc_10']:.4f} (AUC@10), "
              f"{res['Auc_5']:.4f} (AUC@5), {res['Auc_3']:.4f} (AUC@3)")

        def _mean(k):
            return np.mean([v[k] for v in per_category_results.values()])

        print(f"Mean AUC by now: {_mean('Auc_30'):.4f} (AUC@30), {_mean('Auc_15'):.4f} (AUC@15), "
              f"{_mean('Auc_10'):.4f} (AUC@10), {_mean('Auc_5'):.4f} (AUC@5), {_mean('Auc_3'):.4f} (AUC@3)")
        print("=" * 80)

    print("\nSummary of AUC results:")
    print("-" * 50)
    for category in sorted(per_category_results.keys()):
        r = per_category_results[category]
        print(f"{category:<15}: {r['Auc_30']:.4f} (AUC@30), {r['Auc_15']:.4f} (AUC@15), "
              f"{r['Auc_10']:.4f} (AUC@10), {r['Auc_5']:.4f} (AUC@5), {r['Auc_3']:.4f} (AUC@3)")

    if per_category_results:
        def _mean(k):
            return np.mean([v[k] for v in per_category_results.values()])
        print("-" * 50)
        print(f"Mean AUC: {_mean('Auc_30'):.4f} (AUC@30), {_mean('Auc_15'):.4f} (AUC@15), "
              f"{_mean('Auc_10'):.4f} (AUC@10), {_mean('Auc_5'):.4f} (AUC@5), {_mean('Auc_3'):.4f} (AUC@3)")
        print(f"RRA@30={100 * _mean('Rra_30'):.2f} RTA@30={100 * _mean('Rta_30'):.2f}")
        print(f"FINAL_MEAN_AUC AUC30={_mean('Auc_30'):.4f} AUC10={_mean('Auc_10'):.4f} "
              f"AUC5={_mean('Auc_5'):.4f} AUC3={_mean('Auc_3'):.4f} "
              f"RRA30={_mean('Rra_30'):.4f} RTA30={_mean('Rta_30'):.4f}")
    if args.per_sequence_csv and per_sequence_results:
        out_dir = os.path.dirname(os.path.abspath(args.per_sequence_csv))
        os.makedirs(out_dir, exist_ok=True)
        with open(args.per_sequence_csv, "w", newline="") as fout:
            writer = csv.DictWriter(fout, fieldnames=list(per_sequence_results[0]))
            writer.writeheader()
            writer.writerows(per_sequence_results)
        print(f"PER_SEQUENCE_CSV {args.per_sequence_csv} rows={len(per_sequence_results)}")
    print(f"model_path={args.model_path}")


if __name__ == "__main__":
    main()
