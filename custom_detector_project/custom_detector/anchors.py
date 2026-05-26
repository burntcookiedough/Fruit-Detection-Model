"""Generate anchor grid."""
import math

import torch


def generate_anchors(img_size, fm_sizes, anchor_scales, ratios, device='cpu'):
    """Generate anchor boxes for all scales.
    Args:
        img_size: input image size (416)
        fm_sizes: list of feature map sizes, e.g. [208, 104, 52, 26, 13]
        anchor_scales: base anchor size per feature map, e.g. [32, 64, 128, 200, 300]
        ratios: aspect ratios, e.g. [0.5, 1.0, 2.0]
    Returns: anchors [N, 4] in cxcywh format (pixel coords).
    """
    anchors = []
    for fm_size, scale in zip(fm_sizes, anchor_scales):
        stride = img_size / fm_size
        ys = (torch.arange(fm_size, dtype=torch.float32, device=device) + 0.5) * stride
        xs = (torch.arange(fm_size, dtype=torch.float32, device=device) + 0.5) * stride
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        centers = torch.stack([xx, yy], dim=-1).reshape(-1, 2)

        whs = []
        for ratio in ratios:
            w = scale * math.sqrt(ratio)
            h = scale / math.sqrt(ratio)
            whs.append([w, h])
        whs = torch.tensor(whs, dtype=torch.float32, device=device)
        centers = centers[:, None, :].expand(-1, len(ratios), -1)
        whs = whs[None, :, :].expand(centers.shape[0], -1, -1)
        anchors.append(torch.cat([centers, whs], dim=2).reshape(-1, 4))
    return torch.cat(anchors, dim=0)
