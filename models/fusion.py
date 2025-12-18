from __future__ import annotations
import torch
import torch.nn as nn
from einops.layers.torch import Reduce
from .audio import AudioCNNTransformer
from .video import ViTBackbone

class VideoAudioFusion(nn.Module):
    """Cross-attention fusion. Returns logits."""
    def __init__(
        self,
        seq_len_audio: int,
        seq_len_video: int,
        d_hidden_ff_audio: int,
        num_heads_audio: int,
        dropout_ff: float,
        dropout_att: float,
        num_heads_cross: int,
        nb_class: int = 3,
    ):
        super().__init__()
        self.seq_len_audio = seq_len_audio
        self.seq_len_video = seq_len_video

        self.video_backbone = ViTBackbone(trainable=False)
        self.audio_feat = AudioCNNTransformer(seq_len_audio, d_hidden_ff_audio, num_heads_audio, dropout_ff, dropout_att)

        self.cross = nn.MultiheadAttention(1024, num_heads_cross, dropout=0.1, batch_first=True)
        self.reduce = Reduce("b s f -> b f", reduction="mean")
        self.head = nn.Linear(3072, nb_class)

    def forward(self, videos: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        b, fv, c, h, w = videos.shape
        v = self.video_backbone(videos.view(b * fv, c, h, w)).view(b, fv, -1)   # (B, fv, 1024)
        a = self.audio_feat(audio)                                              # (B, fa, 1024)

        if self.seq_len_video != self.seq_len_audio:
            # expected case: fv=75, fa=5 -> pool v into 5 segments
            v = self.reduce(v.view(b * 5, 15, -1)).view(b, 5, -1)

        cross_out, _ = self.cross(a, v, v, need_weights=False)                  # q=a, k=v, v=v
        cat = torch.cat([a, cross_out, v], dim=2)                               # (B, fa, 3072)
        x = self.reduce(cat)
        return self.head(x)

class VideoAudioFusionFlatten(nn.Module):
    def __init__(
        self,
        seq_len_audio: int,
        seq_len_video: int,
        d_hidden_ff_audio: int,
        num_heads_audio: int,
        dropout_ff: float,
        dropout_att: float,
        num_heads_cross: int,
        nb_class: int = 3,
    ):
        super().__init__()
        self.seq_len_audio = seq_len_audio
        self.seq_len_video = seq_len_video

        self.video_backbone = ViTBackbone(trainable=False)
        self.audio_feat = AudioCNNTransformer(seq_len_audio, d_hidden_ff_audio, num_heads_audio, dropout_ff, dropout_att)

        self.cross = nn.MultiheadAttention(1024, num_heads_cross, dropout=0.1, batch_first=True)
        self.reduce = Reduce("b s f -> b f", reduction="mean")
        self.head = nn.Linear(5 * 3072, nb_class)

    def forward(self, videos: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        b, fv, c, h, w = videos.shape
        v = self.video_backbone(videos.view(b * fv, c, h, w)).view(b, fv, -1)
        a = self.audio_feat(audio)

        if self.seq_len_video != self.seq_len_audio:
            v = self.reduce(v.view(b * 5, 15, -1)).view(b, 5, -1)

        cross_out, _ = self.cross(a, v, v, need_weights=False)
        cat = torch.cat([a, cross_out, v], dim=2)  # (B, 5, 3072)

        if cat.size(1) != 5:
            # just in case: collapse into 5 segments
            cat = self.reduce(cat.view(b * 5, 15, -1)).view(b, 5, -1)

        return self.head(cat.reshape(b, -1))
