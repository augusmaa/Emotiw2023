from __future__ import annotations
import torch
import torch.nn as nn
from einops.layers.torch import Reduce
import timm

class Identity(nn.Module):
    def forward(self, x):
        return x

class ViTBackbone(nn.Module):
    def __init__(self, model_name: str = "vit_large_patch14_224_clip_laion2b", pretrained: bool = True, trainable: bool = False):
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=pretrained)
        self.vit.head = Identity()
        if not trainable:
            for p in self.vit.block.parameters():
                p.requires_grad = False

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: (B*F, C, H, W) -> (B*F, 1024)
        return self.vit(frames)

class VideoClassifier(nn.Module):
    def __init__(self, nb_class: int = 3, trainable_backbone: bool = False):
        super().__init__()
        self.backbone = ViTBackbone(trainable=trainable_backbone)
        self.reduce = Reduce("b s f -> b f", reduction="mean")
        self.head = nn.Linear(1024, nb_class)

    def forward(self, videos: torch.Tensor) -> torch.Tensor:
        b, f, c, h, w = videos.shape
        x = videos.view(b * f, c, h, w)
        x = self.backbone(x).view(b, f, -1)
        x = self.reduce(x)
        return self.head(x)

class VideoClassifierFlatten(nn.Module):
    def __init__(self, nb_class: int = 3, trainable_backbone: bool = False):
        super().__init__()
        self.backbone = ViTBackbone(trainable=trainable_backbone)
        self.reduce = Reduce("b s f -> b f", reduction="mean")
        self.head = nn.Linear(5 * 1024, nb_class)

    def forward(self, videos: torch.Tensor) -> torch.Tensor:
        b, f, c, h, w = videos.shape
        x = videos.view(b * f, c, h, w)
        x = self.backbone(x).view(b, f, -1)

        if f == 5:
            return self.head(x.reshape(b, -1))

        x5 = self.reduce(x.view(b * 5, 15, -1)).view(b, 5, -1)
        return self.head(x5.reshape(b, -1))
