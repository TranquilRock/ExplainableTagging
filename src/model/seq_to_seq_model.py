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
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        positional_encoding = torch.zeros(max_len, 1, d_model)
        positional_encoding[:, 0, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('positional_encoding', positional_encoding)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.positional_encoding[:x.size(0)]
        return self.dropout(x)


class SeqtoSeqModel(torch.nn.Module):
    """TODO"""

    def __init__(
        self,
        embeddings: torch.Tensor,
        d_model: int,
        dim_feedforward: int,
        nhead: int,
        num_layers: int,
        dropout: float,
        num_classes: int,
    ) -> None:

        super(SeqtoSeqModel, self).__init__()

        self.embed = nn.Embedding.from_pretrained(embeddings, freeze=True)

        embedding_dim = embeddings.size(1)

        self.prenet = nn.Linear(embedding_dim, d_model)

        self.pos_encoder = PositionalEncoder(d_model, dropout)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, dim_feedforward=dim_feedforward, nhead=nhead, dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers)

        self.pred_layer = nn.Sequential(
            nn.Linear(d_model, num_classes),
            nn.Sigmoid()
        )

    def forward(self, input_tokens):
        embd = self.embed(input_tokens)

        out = self.prenet(embd)
        # out: (length, batch size, d_model)
        out = out.permute(1, 0, 2)
        # Positional encoding
        out = self.pos_encoder(out)
        # The encoder layer expect features in the shape of (length, batch size, d_model).
        out = self.encoder(out)
        # out: (batch size, length, d_model)
        out = out.transpose(0, 1)

        out = self.pred_layer(out)

        return out
