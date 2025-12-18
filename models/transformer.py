from __future__ import annotations
import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """Sin/cos positional encoding (batch_first)."""

    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, F)
        s = x.size(1)
        return x + self.pe[:, :s, :]

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class TransformerEncoderBlock(nn.Module):
    def __init__(self, seq_len: int, d_model: int, d_hidden_ff: int, num_heads: int, dropout_ff: float, dropout_att: float):
        super().__init__()
        self.pe = PositionalEncoding(seq_len, d_model)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout_att, batch_first=True)
        self.ff = FeedForward(d_model, d_hidden_ff, dropout_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, F)
        x = self.pe(x)
        x1 = self.norm1(x)
        att_out, _ = self.attn(x1, x1, x1, need_weights=False)
        x = x + att_out
        x2 = self.norm2(x)
        x = x + self.ff(x2)
        return x
