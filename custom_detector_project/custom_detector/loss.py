"""Focal Loss + SmoothL1."""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .matcher import match_anchors


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets, reduction='sum'):
        """
        logits: [N, C] unnormalized scores
        targets: [N, C] binary one-hot / zero vectors
        """
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        loss = (focal_weight * bce).sum(dim=1)
        if reduction == 'none':
            return loss
        if reduction == 'mean':
            return loss.mean()
        return loss.sum()


class DetectionLoss(nn.Module):
    def __init__(self, num_classes, pos_iou=0.5, neg_iou=0.4, gamma=2.0, alpha=0.25,
                 matcher_type="grid", img_size=416, fm_sizes=None, ratios=None,
                 neg_pos_ratio=100, cache_targets=False, target_cache_dir=None):
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
        self.focal = FocalLoss(gamma, alpha)

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
        cls_loss_total = torch.tensor(0.0, device=device)
        box_loss_total = torch.tensor(0.0, device=device)
        num_pos_total = 0

        for b in range(B):
            gt_boxes = gt_boxes_list[b].to(device)
            gt_labels = gt_labels_list[b].to(device)

            target_key = target_keys[b] if target_keys is not None else None
            cls_targets, box_targets, pos_mask, neg_mask = self._get_targets(
                anchors, gt_boxes, gt_labels, target_key
            )

            valid_mask = pos_mask | neg_mask
            if valid_mask.sum() == 0:
                continue

            # Build binary targets [N, C]
            binary_targets = torch.zeros((anchors.shape[0], num_classes), dtype=torch.float32, device=device)
            binary_targets[pos_mask, cls_targets[pos_mask]] = 1.0

            per_anchor_cls = self.focal(cls_pred[b][valid_mask], binary_targets[valid_mask], reduction='none')
            valid_pos = pos_mask[valid_mask]
            pos_cls = per_anchor_cls[valid_pos].sum()
            neg_cls_all = per_anchor_cls[~valid_pos]
            max_neg = self.neg_pos_ratio * max(int(pos_mask.sum().item()), 1)
            max_neg = min(max_neg, neg_cls_all.numel())
            if max_neg > 0:
                neg_cls = neg_cls_all.topk(max_neg).values.sum()
            else:
                neg_cls = torch.tensor(0.0, device=device)
            cls_loss_total += pos_cls + neg_cls

            if pos_mask.sum() > 0:
                box_loss = F.smooth_l1_loss(box_pred[b][pos_mask], box_targets[pos_mask], reduction='sum')
                box_loss_total += box_loss
                num_pos_total += pos_mask.sum().item()

        if num_pos_total == 0:
            return {'total': cls_loss_total, 'cls': cls_loss_total, 'box': box_loss_total, 'num_pos': 0}

        cls_loss_norm = cls_loss_total / num_pos_total
        box_loss_norm = box_loss_total / num_pos_total
        total = cls_loss_norm + box_loss_norm
        return {'total': total, 'cls': cls_loss_norm, 'box': box_loss_norm, 'num_pos': num_pos_total}
