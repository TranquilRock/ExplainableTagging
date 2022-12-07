"""TODO"""
import math

import torch
import torch.nn as nn


class PositionalEncoder(nn.Module):
    """Layer to insert positional embedding."""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoder, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        positional_encoding = torch.zeros(max_len, 1, d_model)
        positional_encoding[:, 0, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("positional_encoding", positional_encoding)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.positional_encoding[: x.size(0)]
        return self.dropout(x)
