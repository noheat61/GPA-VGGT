"""Ground-truth point clouds for multi-view reconstruction (Acc / Comp / NC).

Two published protocols share this loader because they share a lineage -- both
descend from Spann3R's `eval_recon`, and their dataset code is the same file with
different constants. Keeping one loader here makes the parts that actually differ
explicit, in `frame_ids()` and in `eval_mv_recon.py`:

    protocol   frames per sequence        scored region      point cap
    pi3        a fixed seq-id map,        whole frame,       none
               stride 200/500 (sparse)    GT valid_mask
               or 40/100 (dense)
    litevggt   whole video at stride 3,   centre 224x224     1e6, random
               truncated to a multiple
               of 8

Neither is a superset of the other and their numbers are not comparable: at
stride 500 an NRGBD scene contributes one to four frames, at stride 3 it
contributes 128 to 552. The centre crop matters as much -- it removes the image
periphery, which is exactly where a pose error or a merge artefact lands.

Image and depth geometry is identical in both and is reproduced here rather than
imported, because Pi3-eval's `datasets/` package calls `rootutils.setup_root`
at import time and would put its own `utils/` ahead of ours on sys.path.
Everything below is a transcription of `Pi3-eval/datasets/{nrgbd,sevenscenes}.py`
and `datasets/utils/cropping.py`; the constants are theirs, including the two
hard-coded intrinsics, which LiteVGGT's `eval/data.py` also uses unchanged.

The resize deserves one note: width goes to 518 and height to the nearest
multiple of 14, so 640x480 becomes 518x392. Pi3's source comment says 388 and is
wrong -- 480 * 518/640 = 388.5, which rounds to 28 patches, not 27. 392 is also
what LiteVGGT hard-codes, so the two protocols agree on resolution and differ
only in the three columns above.
"""

import os
import os.path as osp

import cv2
import imageio.v2
import numpy as np
import torch
import torchvision.transforms as tvf
from PIL import Image, ImageFile

from vggt.utils.geometry import (closed_form_inverse_se3,
                                 unproject_depth_map_to_point_map)

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
to_tensor = tvf.ToTensor()

# Hard-coded by both protocols. 7-Scenes is the Kinect factory calibration;
# NRGBD's is the value its own release ships.
INTRINSICS = {
    "sevenscenes": (525.0, 525.0, 320.0, 240.0),
    "nrgbd": (554.2562584220408, 554.2562584220408, 320.0, 240.0),
}


def _resize_image(image, out_wh):
    scale = max(out_wh[0] / image.size[0], out_wh[1] / image.size[1])
    return image.resize(out_wh, resample=(Image.Resampling.LANCZOS if scale < 1
                                          else Image.Resampling.BICUBIC))


def _resize_all(image, depth, intrinsic, output_width):
    """Width to `output_width`, height to the nearest multiple of 14."""
    in_wh = np.array(depth.shape[::-1], dtype=np.float32)
    out_wh = np.array([output_width,
                       round(in_wh[1] * (output_width / in_wh[0]) / 14) * 14])
    image = _resize_image(image, tuple(int(v) for v in out_wh))
    # Nearest, so an invalid pixel is never interpolated into a valid one.
    depth = cv2.resize(depth, tuple(int(v) for v in out_wh),
                       interpolation=cv2.INTER_NEAREST)
    intrinsic = np.copy(intrinsic)
    # Pixel centres: the +0.5 / -0.5 pair makes the scaling exact for a
    # convention where integer coordinates name pixel centres rather than corners.
    intrinsic[0, 2] += 0.5
    intrinsic[1, 2] += 0.5
    intrinsic[:2, :] *= np.max(out_wh / in_wh)
    intrinsic[0, 2] -= 0.5
    intrinsic[1, 2] -= 0.5
    assert image.size == depth.shape[::-1]
    return image, depth, intrinsic


# --------------------------------------------------------------------------
# sequence discovery
# --------------------------------------------------------------------------

def sequences(dataset, root):
    """-> sorted sequence names, the keys the seq-id maps use."""
    if dataset == "nrgbd":
        return sorted(d for d in os.listdir(root) if osp.isdir(osp.join(root, d)))
    if dataset == "sevenscenes":
        out = []
        for scene in sorted(d for d in os.listdir(root)
                            if osp.isdir(osp.join(root, d))):
            split = osp.join(root, scene, "TestSplit.txt")
            if not osp.exists(split):
                continue
            with open(split) as fh:
                out += [f"{scene}/seq-{int(l.strip()[len('sequence'):]):02d}"
                        for l in fh if l.strip()]
        return sorted(out)
    raise SystemExit(f"unknown dataset {dataset}")


def seq_len(dataset, root, seq):
    if dataset == "nrgbd":
        return len(os.listdir(osp.join(root, seq, "images")))
    return len([f for f in os.listdir(osp.join(root, seq))
                if f.endswith(".color.png")])


def frame_ids(dataset, root, seq, protocol, kf, seq_id_map=None):
    """Which frames of `seq` the protocol scores.

    `pi3` reads the published map so a rerun cannot drift; `litevggt` derives
    the list, because its rule is entirely in the dataset class (stride, then
    truncate to a multiple of 8 -- `LiteVGGT/eval/data.py:119` and `:284`).
    """
    if protocol == "pi3":
        if seq not in seq_id_map:
            return None
        return list(seq_id_map[seq])
    n = seq_len(dataset, root, seq)
    # NRGBD's own line is `[:: min(kf_every, len(idxs) // 2)]`, which only bites
    # on a sequence shorter than 2*kf. Reproduced so a short scene behaves the
    # same way it does upstream.
    step = kf if dataset == "sevenscenes" else min(kf, max(n // 2, 1))
    ids = list(range(n))[::step]
    if len(ids) % 8:
        ids = ids[:len(ids) - (len(ids) % 8)]
    return ids


# --------------------------------------------------------------------------
# per-sequence data
# --------------------------------------------------------------------------

def _nrgbd_extrinsics(root, seq):
    """poses.txt is 4 lines per c2w matrix, in the OpenGL convention."""
    with open(osp.join(root, seq, "poses.txt")) as fh:
        raw = [l.strip() for l in fh if l.strip()]
    poses = np.array([[[float(x) for x in raw[i + k].split()] for k in range(4)]
                      for i in range(0, len(raw), 4)], dtype=np.float32)
    poses[:, :, 1:3] *= -1.0                       # gl -> cv
    return closed_form_inverse_se3(poses)[:, :3, :]


def load(dataset, root, seq, ids, load_img_size=518):
    """-> dict(image_paths, images (N,3,H,W), pointclouds (N,H,W,3), valid_mask).

    `pointclouds` are world-frame ground truth, unprojected from the ground-truth
    depth with the ground-truth pose -- so they carry the dataset's own scale and
    the prediction has to be aligned to them, not the other way round.
    """
    fx, fy, cx, cy = INTRINSICS[dataset]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    intrinsics = np.tile(K, (len(ids), 1, 1))

    if dataset == "nrgbd":
        extrinsics = _nrgbd_extrinsics(root, seq)[ids]
    else:
        extrinsics = np.zeros((len(ids), 3, 4), dtype=np.float32)

    paths, images, depths = [], [], []
    for k, i in enumerate(ids):
        if dataset == "nrgbd":
            impath = osp.join(root, seq, "images", f"img{i}.png")
            dpath = osp.join(root, seq, "depth", f"depth{i}.png")
        else:
            impath = osp.join(root, seq, f"frame-{i:06d}.color.png")
            dpath = osp.join(root, seq, f"frame-{i:06d}.depth.proj.png")
            extrinsics[k] = closed_form_inverse_se3(
                np.loadtxt(osp.join(root, seq, f"frame-{i:06d}.pose.txt"))[None]
            )[0][:3, :]

        rgb = Image.open(impath)
        depth = imageio.v2.imread(dpath)
        assert depth.shape == (480, 640), f"{dpath}: depth is {depth.shape}"
        rgb = _resize_image(rgb, (depth.shape[1], depth.shape[0]))
        if dataset == "sevenscenes":
            depth = depth.copy()
            depth[depth == 65535] = 0          # 7-Scenes' invalid marker
        depth = np.nan_to_num(depth.astype(np.float32), 0.0) / 1000.0
        depth[depth > 10] = 0                  # too far
        depth[depth < 1e-3] = 0                # too near
        rgb, depth, intrinsics[k] = _resize_all(rgb, depth, intrinsics[k],
                                                load_img_size)
        paths.append(impath)
        images.append(to_tensor(rgb))
        depths.append(depth)

    depths = np.array(depths)
    pointclouds = unproject_depth_map_to_point_map(
        depth_map=depths[..., None], intrinsics_cam=intrinsics,
        extrinsics_cam=extrinsics)
    return {"seq": seq, "image_paths": paths,
            "images": torch.stack(images, dim=0),
            "pointclouds": pointclouds, "valid_mask": depths > 1e-4}
