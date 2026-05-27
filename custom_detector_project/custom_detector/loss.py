"""Focal Loss + CIoU box regression."""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .matcher import match_anchors
from .utils import cxcywh_to_xyxy


def compute_class_weights(label_dir, num_classes, smoothing=0.1):
    """Compute inverse-frequency class weights from YOLO label directory.

    Returns a tensor of shape [num_classes] where rare classes get higher weight.
    Uses sqrt-inverse-frequency with Laplace smoothing for stability.
    """
    from collections import Counter
    counts = Counter()
    for fname in os.listdir(label_dir):
        if not fname.endswith('.txt'):
            continue
        with open(os.path.join(label_dir, fname)) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    counts[int(parts[0])] += 1
    total = sum(counts.values())
    if total == 0:
        return torch.ones(num_classes)
    freq = torch.tensor([counts.get(i, 0) + smoothing for i in range(num_classes)])
    # sqrt-inverse-frequency: less aggressive than pure inverse, avoids overfitting rare classes
    weights = torch.sqrt(freq.sum() / (num_classes * freq))
    # Normalize so mean weight = 1.0
    weights = weights / weights.mean()
    return weights


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, class_weights=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        # class_weights: optional [C] tensor to upweight rare classes
        self.register_buffer('class_weights',
                             class_weights if class_weights is not None else None)

    def forward(self, logits, targets, reduction='sum', class_ids=None):
        """
        logits: [N, C] unnormalized scores
        targets: [N, C] binary one-hot / zero vectors
        class_ids: optional [N] int tensor — GT class per anchor (for weighting positives)
        """
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        loss = (focal_weight * bce).sum(dim=1)

        # Apply per-class weights to positive anchors
        if self.class_weights is not None and class_ids is not None:
            # class_ids = -1 for negatives, 0..C-1 for positives
            is_pos = class_ids >= 0
            per_anchor_weight = torch.ones_like(loss)
            if is_pos.any():
                cw = self.class_weights.to(loss.device)
                per_anchor_weight[is_pos] = cw[class_ids[is_pos]]
            loss = loss * per_anchor_weight

        if reduction == 'none':
            return loss
        if reduction == 'mean':
            return loss.mean()
        return loss.sum()


def _decode_box_pred(box_pred, anchors):
    """Decode predicted offsets (tx, ty, tw, th) back to absolute cxcywh boxes."""
    tx, ty, tw, th = box_pred[:, 0], box_pred[:, 1], box_pred[:, 2], box_pred[:, 3]
    tw = tw.clamp(min=-8.0, max=8.0)
    th = th.clamp(min=-8.0, max=8.0)
    cx = tx * anchors[:, 2] + anchors[:, 0]
    cy = ty * anchors[:, 3] + anchors[:, 1]
    w = torch.exp(tw) * anchors[:, 2]
    h = torch.exp(th) * anchors[:, 3]
    return torch.stack([cx, cy, w, h], dim=1)


@torch.amp.autocast('cuda', enabled=False)
def ciou_loss(pred_cxcywh, target_cxcywh, img_size):
    """Complete IoU loss between predicted and target boxes (both in cxcywh).

    CIoU = IoU - (distance^2 / diagonal^2) - alpha * v
    where v measures aspect ratio consistency.
    Returns per-box loss = 1 - CIoU, clamped to [0, 4].
    """
    # Ensure float32 for numerical stability
    pred_cxcywh = pred_cxcywh.float()
    target_cxcywh = target_cxcywh.float()
    # Convert to xyxy for IoU computation
    pred_xyxy = cxcywh_to_xyxy(pred_cxcywh).clamp(min=0, max=img_size)
    target_xyxy = cxcywh_to_xyxy(target_cxcywh).clamp(min=0, max=img_size)

    # Intersection
    inter_x1 = torch.max(pred_xyxy[:, 0], target_xyxy[:, 0])
    inter_y1 = torch.max(pred_xyxy[:, 1], target_xyxy[:, 1])
    inter_x2 = torch.min(pred_xyxy[:, 2], target_xyxy[:, 2])
    inter_y2 = torch.min(pred_xyxy[:, 3], target_xyxy[:, 3])
    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    # Union
    pred_area = (pred_xyxy[:, 2] - pred_xyxy[:, 0]) * (pred_xyxy[:, 3] - pred_xyxy[:, 1])
    target_area = (target_xyxy[:, 2] - target_xyxy[:, 0]) * (target_xyxy[:, 3] - target_xyxy[:, 1])
    union_area = pred_area + target_area - inter_area + 1e-7

    iou = inter_area / union_area

    # Enclosing box diagonal
    enclose_x1 = torch.min(pred_xyxy[:, 0], target_xyxy[:, 0])
    enclose_y1 = torch.min(pred_xyxy[:, 1], target_xyxy[:, 1])
    enclose_x2 = torch.max(pred_xyxy[:, 2], target_xyxy[:, 2])
    enclose_y2 = torch.max(pred_xyxy[:, 3], target_xyxy[:, 3])
    enclose_diag_sq = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2 + 1e-7

    # Center distance
    center_dist_sq = (pred_cxcywh[:, 0] - target_cxcywh[:, 0]) ** 2 + \
                     (pred_cxcywh[:, 1] - target_cxcywh[:, 1]) ** 2

    # Aspect ratio penalty
    pred_w = pred_cxcywh[:, 2].clamp(min=1e-6)
    pred_h = pred_cxcywh[:, 3].clamp(min=1e-6)
    target_w = target_cxcywh[:, 2].clamp(min=1e-6)
    target_h = target_cxcywh[:, 3].clamp(min=1e-6)
    v = (4.0 / (math.pi ** 2)) * (torch.atan(target_w / target_h) - torch.atan(pred_w / pred_h)) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + 1e-7)

    ciou = iou - center_dist_sq / enclose_diag_sq - alpha * v
    return (1 - ciou).clamp(min=0, max=4.0)


class DetectionLoss(nn.Module):
    def __init__(self, num_classes, pos_iou=0.5, neg_iou=0.4, gamma=2.0, alpha=0.25,
                 matcher_type="grid", img_size=416, fm_sizes=None, ratios=None,
                 neg_pos_ratio=100, cache_targets=False, target_cache_dir=None,
                 class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.pos_iou = pos_iou
        self.neg_iou = neg_iou
        self.matcher_type = matcher_type
        self.img_size = img_size
        self.fm_sizes = fm_sizes
        self.ratios = ratios
        self.neg_pos_ratio = neg_pos_ratio
        self.cache_targets = cache_targets
        self.target_cache_dir = target_cache_dir
        self.focal = FocalLoss(gamma, alpha, class_weights=class_weights)

    def _cache_path(self, key):
        if not self.target_cache_dir or key is None:
            return None
        return os.path.join(self.target_cache_dir, f"{key}.pt")

    def _targets_from_compact_cache(self, path, num_anchors, device):
        cached = torch.load(path, map_location=device)
        cls_targets = torch.full((num_anchors,), -1, dtype=torch.long, device=device)
        box_targets = torch.zeros((num_anchors, 4), dtype=torch.float32, device=device)
        pos_mask = torch.zeros(num_anchors, dtype=torch.bool, device=device)
        neg_mask = torch.ones(num_anchors, dtype=torch.bool, device=device)

        pos_idx = cached["pos_idx"].to(device)
        if pos_idx.numel() > 0:
            pos_mask[pos_idx] = True
            neg_mask[pos_idx] = False
            cls_targets[pos_idx] = cached["cls_pos"].to(device)
            box_targets[pos_idx] = cached["box_pos"].to(device)
        return cls_targets, box_targets, pos_mask, neg_mask

    def _save_compact_cache(self, path, cls_targets, box_targets, pos_mask):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pos_idx = pos_mask.nonzero(as_tuple=False).squeeze(1).cpu()
        payload = {
            "pos_idx": pos_idx,
            "cls_pos": cls_targets[pos_mask].cpu(),
            "box_pos": box_targets[pos_mask].cpu(),
        }
        tmp_path = path + ".tmp"
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)

    def _get_targets(self, anchors, gt_boxes, gt_labels, target_key=None):
        cache_path = self._cache_path(target_key)
        if self.cache_targets and cache_path and os.path.exists(cache_path):
            return self._targets_from_compact_cache(cache_path, anchors.shape[0], anchors.device)

        targets = match_anchors(
            anchors, gt_boxes, gt_labels, self.pos_iou, self.neg_iou,
            mode=self.matcher_type, img_size=self.img_size,
            fm_sizes=self.fm_sizes, ratios=self.ratios
        )
        if self.cache_targets and cache_path:
            self._save_compact_cache(cache_path, targets[0], targets[1], targets[2])
        return targets

    def forward(self, cls_pred, box_pred, anchors, gt_boxes_list, gt_labels_list, target_keys=None):
        """
        cls_pred: [B, N, C]
        box_pred: [B, N, 4]
        anchors: [N, 4]
        """
        B = cls_pred.shape[0]
        device = cls_pred.device
        num_classes = cls_pred.shape[2]
        num_anchors = anchors.shape[0]

        # Phase 1: Compute all targets (still per-image due to variable GT count)
        all_cls_targets = []
        all_box_targets = []
        all_pos_masks = []
        all_neg_masks = []

        for b in range(B):
            gt_boxes = gt_boxes_list[b].to(device)
            gt_labels = gt_labels_list[b].to(device)
            target_key = target_keys[b] if target_keys is not None else None
            ct, bt, pm, nm = self._get_targets(anchors, gt_boxes, gt_labels, target_key)
            all_cls_targets.append(ct)
            all_box_targets.append(bt)
            all_pos_masks.append(pm)
            all_neg_masks.append(nm)

        # Stack into batch tensors [B, N] and [B, N, 4]
        cls_targets = torch.stack(all_cls_targets)
        box_targets = torch.stack(all_box_targets)
        pos_mask = torch.stack(all_pos_masks)
        neg_mask = torch.stack(all_neg_masks)

        # Phase 2: Batched classification loss
        valid_mask = pos_mask | neg_mask  # [B, N]

        # Build binary targets [B, N, C] — vectorized one-hot
        binary_targets = torch.zeros(B, num_anchors, num_classes, dtype=torch.float32, device=device)
        pos_b, pos_n = pos_mask.nonzero(as_tuple=True)
        if pos_b.numel() > 0:
            binary_targets[pos_b, pos_n, cls_targets[pos_b, pos_n]] = 1.0

        # Flatten valid entries and compute focal loss in one call
        valid_cls_pred = cls_pred[valid_mask]        # [V, C]
        valid_bt = binary_targets[valid_mask]          # [V, C]
        valid_class_ids = cls_targets[valid_mask]      # [V]

        if valid_cls_pred.numel() == 0:
            zero = torch.tensor(0.0, device=device)
            return {'total': zero, 'cls': zero, 'box': zero, 'num_pos': 0}

        per_anchor_cls = self.focal(valid_cls_pred, valid_bt, reduction='none', class_ids=valid_class_ids)

        # Separate positive and negative losses
        valid_pos_flat = pos_mask[valid_mask]  # [V] bool
        pos_cls = per_anchor_cls[valid_pos_flat].sum()

        if self.neg_pos_ratio <= 0:
            neg_cls = per_anchor_cls[~valid_pos_flat].sum()
        else:
            neg_cls_all = per_anchor_cls[~valid_pos_flat]
            num_pos_total_for_ratio = max(int(pos_mask.sum().item()), 1)
            max_neg = self.neg_pos_ratio * num_pos_total_for_ratio
            max_neg = min(max_neg, neg_cls_all.numel())
            if max_neg > 0:
                neg_cls = neg_cls_all.topk(max_neg).values.sum()
            else:
                neg_cls = torch.tensor(0.0, device=device)

        cls_loss_total = pos_cls + neg_cls

        # Phase 3: Batched box regression loss (CIoU)
        num_pos_total = pos_mask.sum().item()
        if num_pos_total > 0:
            # Expand anchors to [B, N, 4] for batched indexing
            anchors_exp = anchors.unsqueeze(0).expand(B, -1, -1)
            pred_boxes_abs = _decode_box_pred(box_pred[pos_mask], anchors_exp[pos_mask])
            target_boxes_abs = _decode_box_pred(box_targets[pos_mask], anchors_exp[pos_mask])
            box_loss_total = ciou_loss(pred_boxes_abs, target_boxes_abs, self.img_size).sum()
        else:
            box_loss_total = torch.tensor(0.0, device=device)

        cls_loss_norm = cls_loss_total / max(num_pos_total, 1)
        box_loss_norm = box_loss_total / max(num_pos_total, 1)
        total = cls_loss_norm + box_loss_norm
        return {'total': total, 'cls': cls_loss_norm, 'box': box_loss_norm, 'num_pos': num_pos_total}
