"""Relative-pose-error / AUC metrics for the CO3D pose benchmark.

Rotation/translation angle errors are computed over all unordered frame pairs of
a sequence, so the metric is invariant to the global gauge that VGGT fixes at the
first camera. Depends only on `vggt.utils`, so it stays valid for any merging or
attention configuration.
"""

import numpy as np
import torch

from vggt.utils.rotation import mat_to_quat
from vggt.utils.geometry import closed_form_inverse_se3


def convert_pt3d_RT_to_opencv(Rot, Trans):
    """Convert PyTorch3D extrinsics (CO3D annotation format) to an OpenCV 3x4 w2c matrix."""
    rot_pt3d = np.array(Rot)
    trans_pt3d = np.array(Trans)

    trans_pt3d[:2] *= -1
    rot_pt3d[:, :2] *= -1
    rot_pt3d = rot_pt3d.transpose(1, 0)
    extri_opencv = np.hstack((rot_pt3d, trans_pt3d[:, None]))
    return extri_opencv


def build_pair_index(N, B=1):
    """Indices for all unordered frame pairs."""
    i1_, i2_ = torch.combinations(torch.arange(N), 2, with_replacement=False).unbind(-1)
    i1, i2 = [(i[None] + torch.arange(B)[:, None] * N).reshape(-1) for i in [i1_, i2_]]
    return i1, i2


def rotation_angle(rot_gt, rot_pred, batch_size=None, eps=1e-15):
    """Rotation angle error (degrees) between GT and predicted rotations."""
    q_pred = mat_to_quat(rot_pred)
    q_gt = mat_to_quat(rot_gt)

    loss_q = (1 - (q_pred * q_gt).sum(dim=1) ** 2).clamp(min=eps)
    err_q = torch.arccos(1 - 2 * loss_q)

    rel_rangle_deg = err_q * 180 / np.pi
    if batch_size is not None:
        rel_rangle_deg = rel_rangle_deg.reshape(batch_size, -1)
    return rel_rangle_deg


def compare_translation_by_angle(t_gt, t, eps=1e-15, default_err=1e6):
    """Angle (radians) between normalized translation vectors."""
    t_norm = torch.norm(t, dim=1, keepdim=True)
    t = t / (t_norm + eps)

    t_gt_norm = torch.norm(t_gt, dim=1, keepdim=True)
    t_gt = t_gt / (t_gt_norm + eps)

    loss_t = torch.clamp_min(1.0 - torch.sum(t * t_gt, dim=1) ** 2, eps)
    err_t = torch.acos(torch.sqrt(1 - loss_t))

    err_t[torch.isnan(err_t) | torch.isinf(err_t)] = default_err
    return err_t


def translation_angle(tvec_gt, tvec_pred, batch_size=None, ambiguity=True):
    """Translation angle error (degrees) between GT and predicted translations."""
    rel_tangle_deg = compare_translation_by_angle(tvec_gt, tvec_pred)
    rel_tangle_deg = rel_tangle_deg * 180.0 / np.pi

    if ambiguity:
        rel_tangle_deg = torch.min(rel_tangle_deg, (180 - rel_tangle_deg).abs())
    if batch_size is not None:
        rel_tangle_deg = rel_tangle_deg.reshape(batch_size, -1)
    return rel_tangle_deg


def calculate_auc_np(r_error, t_error, max_threshold=30):
    """AUC of the max(r_error, t_error) cumulative histogram up to max_threshold."""
    error_matrix = np.concatenate((r_error[:, None], t_error[:, None]), axis=1)
    max_errors = np.max(error_matrix, axis=1)
    bins = np.arange(max_threshold + 1)
    histogram, _ = np.histogram(max_errors, bins=bins)
    num_pairs = float(len(max_errors))
    normalized_histogram = histogram.astype(float) / num_pairs
    return np.mean(np.cumsum(normalized_histogram)), normalized_histogram


def se3_to_relative_pose_error(pred_se3, gt_se3, num_frames):
    """Rotation/translation angle errors over all frame pairs (w2c poses)."""
    pair_idx_i1, pair_idx_i2 = build_pair_index(num_frames)

    relative_pose_gt = gt_se3[pair_idx_i1].bmm(
        closed_form_inverse_se3(gt_se3[pair_idx_i2])
    )
    relative_pose_pred = pred_se3[pair_idx_i1].bmm(
        closed_form_inverse_se3(pred_se3[pair_idx_i2])
    )

    rel_rangle_deg = rotation_angle(
        relative_pose_gt[:, :3, :3], relative_pose_pred[:, :3, :3]
    )
    rel_tangle_deg = translation_angle(
        relative_pose_gt[:, :3, 3], relative_pose_pred[:, :3, 3]
    )
    return rel_rangle_deg, rel_tangle_deg
