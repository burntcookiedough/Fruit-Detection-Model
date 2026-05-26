"""Assign GT boxes to anchors."""
import torch
import torchvision.ops as ops
from .utils import cxcywh_to_xyxy


def _encode_offsets(anchors, gt_boxes):
    tx = (gt_boxes[:, 0] - anchors[:, 0]) / anchors[:, 2].clamp(min=1e-6)
    ty = (gt_boxes[:, 1] - anchors[:, 1]) / anchors[:, 3].clamp(min=1e-6)
    tw = torch.log(gt_boxes[:, 2].clamp(min=1e-6) / anchors[:, 2].clamp(min=1e-6))
    th = torch.log(gt_boxes[:, 3].clamp(min=1e-6) / anchors[:, 3].clamp(min=1e-6))
    return torch.stack([tx, ty, tw, th], dim=1)


def match_anchors_iou(anchors, gt_boxes, gt_labels, pos_iou=0.5, neg_iou=0.4):
    """
    Args:
        anchors: [N, 4] cxcywh
        gt_boxes: [M, 4] cxcywh
        gt_labels: [M]
    Returns:
        cls_targets: [N] long, -1 = ignore/background, 0..C-1 = class
        box_targets: [N, 4] cxcywh offsets, zero for non-positive
        pos_mask: [N] bool
        neg_mask: [N] bool
    """
    num_anchors = anchors.shape[0]

    cls_targets = torch.full((num_anchors,), -1, dtype=torch.long, device=anchors.device)
    box_targets = torch.zeros((num_anchors, 4), dtype=torch.float32, device=anchors.device)
    pos_mask = torch.zeros(num_anchors, dtype=torch.bool, device=anchors.device)
    neg_mask = torch.zeros(num_anchors, dtype=torch.bool, device=anchors.device)

    if gt_boxes.numel() == 0:
        neg_mask[:] = True
        return cls_targets, box_targets, pos_mask, neg_mask

    anchors_xyxy = cxcywh_to_xyxy(anchors)
    gt_xyxy = cxcywh_to_xyxy(gt_boxes)
    iou = ops.box_iou(anchors_xyxy, gt_xyxy)  # [N, M]

    best_gt_iou, best_gt_idx = iou.max(dim=1)

    pos_mask = best_gt_iou >= pos_iou
    neg_mask = best_gt_iou < neg_iou

    # Force best anchor for each GT to be positive
    best_anchor_for_gt = iou.max(dim=0)[1]
    pos_mask[best_anchor_for_gt] = True
    neg_mask[best_anchor_for_gt] = False

    cls_targets[pos_mask] = gt_labels[best_gt_idx[pos_mask]]

    matched_gt = gt_boxes[best_gt_idx]
    pos_anchors = anchors[pos_mask]
    pos_gt = matched_gt[pos_mask]

    box_targets[pos_mask] = _encode_offsets(pos_anchors, pos_gt)

    return cls_targets, box_targets, pos_mask, neg_mask


def match_anchors_grid(anchors, gt_boxes, gt_labels, img_size=416, fm_sizes=None, ratios=None):
    """Fast YOLO-style assignment.

    Each GT is assigned to the best-ratio anchor in the grid cell containing
    its centre on every feature scale. This avoids the full N x M IoU matrix
    while still giving every object supervised anchors at multiple scales.
    """
    if fm_sizes is None:
        fm_sizes = [208, 104, 52, 26, 13]
    if ratios is None:
        ratios = [0.5, 1.0, 2.0]

    num_anchors = anchors.shape[0]
    cls_targets = torch.full((num_anchors,), -1, dtype=torch.long, device=anchors.device)
    box_targets = torch.zeros((num_anchors, 4), dtype=torch.float32, device=anchors.device)
    pos_mask = torch.zeros(num_anchors, dtype=torch.bool, device=anchors.device)
    neg_mask = torch.ones(num_anchors, dtype=torch.bool, device=anchors.device)

    if gt_boxes.numel() == 0:
        return cls_targets, box_targets, pos_mask, neg_mask

    offset = 0
    candidate_indices = []
    num_ratios = len(ratios)
    for fm_size in fm_sizes:
        stride = img_size / fm_size
        gx = torch.clamp((gt_boxes[:, 0] / stride).long(), min=0, max=fm_size - 1)
        gy = torch.clamp((gt_boxes[:, 1] / stride).long(), min=0, max=fm_size - 1)
        cell_base = offset + (gy * fm_size + gx) * num_ratios
        per_ratio = cell_base[:, None] + torch.arange(num_ratios, device=anchors.device)

        cand_anchors = anchors[per_ratio.reshape(-1)].reshape(gt_boxes.shape[0], num_ratios, 4)
        gt_wh = gt_boxes[:, None, 2:4].expand(-1, num_ratios, -1)
        anchor_wh = cand_anchors[:, :, 2:4]
        inter = torch.minimum(gt_wh[..., 0], anchor_wh[..., 0]) * torch.minimum(gt_wh[..., 1], anchor_wh[..., 1])
        union = gt_wh[..., 0] * gt_wh[..., 1] + anchor_wh[..., 0] * anchor_wh[..., 1] - inter
        best_ratio = (inter / union.clamp(min=1e-6)).argmax(dim=1)
        candidate_indices.append(per_ratio[torch.arange(gt_boxes.shape[0], device=anchors.device), best_ratio])

        offset += fm_size * fm_size * num_ratios

    pos_idx = torch.cat(candidate_indices)
    gt_idx = torch.arange(gt_boxes.shape[0], device=anchors.device).repeat(len(fm_sizes))

    pos_mask[pos_idx] = True
    neg_mask[pos_idx] = False
    cls_targets[pos_idx] = gt_labels[gt_idx]
    box_targets[pos_idx] = _encode_offsets(anchors[pos_idx], gt_boxes[gt_idx])

    return cls_targets, box_targets, pos_mask, neg_mask


def match_anchors(anchors, gt_boxes, gt_labels, pos_iou=0.5, neg_iou=0.4, mode="grid",
                  img_size=416, fm_sizes=None, ratios=None):
    if mode == "iou":
        return match_anchors_iou(anchors, gt_boxes, gt_labels, pos_iou, neg_iou)
    if mode == "grid":
        return match_anchors_grid(anchors, gt_boxes, gt_labels, img_size, fm_sizes, ratios)
    raise ValueError(f"Unknown matcher mode: {mode}")
