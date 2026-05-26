"""Backbone + detection heads."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .anchors import generate_anchors


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        return x


class DetectionHead(nn.Module):
    def __init__(self, in_ch, num_classes, num_ratios):
        super().__init__()
        self.num_classes = num_classes
        self.num_ratios = num_ratios
        mid = max(in_ch // 2, 64)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )
        self.cls_head = nn.Conv2d(mid, num_ratios * num_classes, 3, padding=1)
        self.box_head = nn.Conv2d(mid, num_ratios * 4, 3, padding=1)
        nn.init.constant_(self.cls_head.bias, -4.595)
        nn.init.zeros_(self.box_head.weight)
        nn.init.zeros_(self.box_head.bias)

    def forward(self, x):
        B, _, H, W = x.shape
        x = self.conv(x)
        cls = self.cls_head(x)
        box = self.box_head(x)
        cls = cls.permute(0, 2, 3, 1).reshape(B, H * W * self.num_ratios, self.num_classes)
        box = box.permute(0, 2, 3, 1).reshape(B, H * W * self.num_ratios, 4)
        return cls, box


class FruitDetector(nn.Module):
    def __init__(self, num_classes, img_size, anchor_scales, ratios, fm_sizes=None):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.anchor_scales = anchor_scales
        self.ratios = ratios
        self.all_fm_sizes = [img_size // 2, img_size // 4, img_size // 8, img_size // 16, img_size // 32]
        self.fm_sizes = fm_sizes or [img_size // 8, img_size // 16, img_size // 32]

        self.backbone = nn.ModuleList([
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 256),
        ])

        channels_by_fm = dict(zip(self.all_fm_sizes, [32, 64, 128, 256, 256]))
        if len(self.fm_sizes) != len(anchor_scales):
            raise ValueError("fm_sizes and anchor_scales must have the same length")
        missing = [fm for fm in self.fm_sizes if fm not in channels_by_fm]
        if missing:
            raise ValueError(f"Unsupported feature map sizes for img_size={img_size}: {missing}")

        self.heads = nn.ModuleDict({
            str(fm): DetectionHead(channels_by_fm[fm], num_classes, len(ratios))
            for fm in self.fm_sizes
        })

        anchors = generate_anchors(img_size, self.fm_sizes, anchor_scales, ratios)
        self.register_buffer('anchors', anchors)

    def forward(self, x):
        cls_outs = []
        box_outs = []
        for block, fm_size in zip(self.backbone, self.all_fm_sizes):
            x = block(x)
            fm_key = str(fm_size)
            if fm_key in self.heads:
                cls, box = self.heads[fm_key](x)
                cls_outs.append(cls)
                box_outs.append(box)
        cls_all = torch.cat(cls_outs, dim=1)
        box_all = torch.cat(box_outs, dim=1)
        return cls_all, box_all, self.anchors
