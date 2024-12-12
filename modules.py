import torch.nn as nn
import torch.nn.functional as F
import torch

class DWUnit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DWUnit, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.conv3x3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1x1(x)
        x = self.conv3x3(x)
        x = self.bn(x)
        return self.relu(x)

class DWBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DWBlock, self).__init__()
        self.dw1 = DWUnit(in_channels, out_channels)
        self.dw2 = DWUnit(out_channels, out_channels)

    def forward(self, x):
        x = self.dw1(x)
        return self.dw2(x)

class DecoupledHead(nn.Module):
    def __init__(self, in_channels, num_classes=1):
        super(DecoupledHead, self).__init__()
        self.cls_head = DWUnit(in_channels, num_classes)
        self.obj_head = DWUnit(in_channels, 1)
        self.bbox_head = DWUnit(in_channels, 4)

    def forward(self, x):
        cls_out = self.cls_head(x)
        obj_out = self.obj_head(x)
        bbox_out = self.bbox_head(x)
        return cls_out, obj_out, bbox_out

class YuNetBackbone(nn.Module):
    def __init__(self):
        super(YuNetBackbone, self).__init__()
        self.stage0 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            DWBlock(16, 16),
        )
        self.stage1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DWBlock(16, 16),
            DWBlock(16, 64),
        )
        self.stage2 = nn.Sequential(
            DWBlock(64, 64),
            nn.MaxPool2d(kernel_size=2),
        )
        self.stage3 = nn.Sequential(
            DWBlock(64, 64),
            nn.MaxPool2d(kernel_size=2),
        )
        self.stage4 = nn.Sequential(
            DWBlock(64, 64),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x):
        c1 = self.stage0(x)
        c2 = self.stage1(c1)
        c3 = self.stage2(c2)
        c4 = self.stage3(c3)
        c5 = self.stage4(c4)
        return c3, c4, c5
    
class TFPN(nn.Module):
    def __init__(self):
        super(TFPN, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.fpn1 = DWUnit(64, 64)
        self.fpn2 = DWUnit(64, 64)
        self.fpn3 = DWUnit(64, 64)

    def forward(self, c3, c4, c5):
        p5 = self.fpn1(c5)
        p4 = self.upsample(p5) + self.fpn2(c4)
        p3 = self.upsample(p4) + self.fpn3(c3)
        return p3, p4, p5
    
class YuNet(nn.Module):
    def __init__(self):
        super(YuNet, self).__init__()
        self.backbone = YuNetBackbone()
        self.fpn = TFPN()
        self.heads = nn.ModuleList([
            nn.Sequential(
                DWUnit(64, 64),
                DecoupledHead(64)
            ) for _ in range(3)
        ])

    def forward(self, x):
        c3, c4, c5 = self.backbone(x)
        ps = self.fpn(c3, c4, c5)
        cls_preds, obj_preds, bbox_preds = [], [], []
        for i, p in enumerate(ps):
            cls_pred, obj_pred, bbox_pred = self.heads[i](p)
            cls_pred = cls_pred.permute(0, 2, 3, 1).reshape(x.size(0), -1, cls_pred.size(1))
            obj_pred = obj_pred.permute(0, 2, 3, 1).reshape(x.size(0), -1, 1)
            bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(x.size(0), -1, 4)
            cls_preds.append(cls_pred)
            obj_preds.append(obj_pred)
            bbox_preds.append(bbox_pred)

        cls_preds = torch.cat(cls_preds, dim=1)
        obj_preds = torch.cat(obj_preds, dim=1)
        bbox_preds = torch.cat(bbox_preds, dim=1)
        return cls_preds, obj_preds, bbox_preds