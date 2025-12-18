from __future__ import annotations
import torch
import torch.nn as nn
from einops.layers.torch import Reduce
from .transformer import TransformerEncoderBlock

class AudioCNNBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 32, 3), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(64, 128, 3), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.cnn3 = nn.Sequential(
            nn.Conv2d(128, 256, 3), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.cnn4 = nn.Sequential(
            nn.Conv2d(256, 512, 3), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*F, 1, H, W)
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.cnn4(x)
        x = self.avgpool(x)
        return x

class AudioCNNTransformer(nn.Module):
    """Returns per-frame embeddings: (B, F, 1024)"""
    def __init__(self, seq_len: int, d_hidden_ff: int, num_heads: int, dropout_ff: float, dropout_att: float, d_model: int = 1024):
        super().__init__()
        self.backbone = AudioCNNBackbone()
        self.encoder = TransformerEncoderBlock(seq_len, d_model, d_hidden_ff, num_heads, dropout_ff, dropout_att)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        # spec: (B, F, 1, H, W)
        b, f, c, h, w = spec.shape
        x = spec.view(b * f, c, h, w)
        x = self.backbone(x)
        x = x.view(b, f, -1)
        x = self.encoder(x)
        return x

class AudioClassifier(nn.Module):
    def __init__(self, seq_len: int, d_hidden_ff: int, num_heads: int, dropout_ff: float, dropout_att: float, nb_class: int = 3):
        super().__init__()
        self.feat = AudioCNNTransformer(seq_len, d_hidden_ff, num_heads, dropout_ff, dropout_att)
        self.reduce = Reduce("b s f -> b f", reduction="mean")
        self.head = nn.Linear(1024, nb_class)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        x = self.feat(spec)
        x = self.reduce(x)
        return self.head(x)  # logits

class AudioClassifierFlatten(nn.Module):
    """Flatten 5 segments into one vector: (B, 5*1024) -> logits"""
    def __init__(self, seq_len: int, d_hidden_ff: int, num_heads: int, dropout_ff: float, dropout_att: float, nb_class: int = 3):
        super().__init__()
        self.feat = AudioCNNTransformer(seq_len, d_hidden_ff, num_heads, dropout_ff, dropout_att)
        self.reduce = Reduce("b s f -> b f", reduction="mean")
        self.head = nn.Linear(5 * 1024, nb_class)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        b, fr, *_ = spec.shape
        x = self.feat(spec)  # (B, fr, 1024)

        if fr == 5:
            return self.head(x.reshape(b, -1))

        # fr==75 -> reduce into 5 segments of 15 frames
        x5 = self.reduce(x.view(b * 5, 15, -1)).view(b, 5, -1)
        return self.head(x5.reshape(b, -1))
