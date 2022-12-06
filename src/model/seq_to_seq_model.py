"""TODO"""

import torch
import torch.nn as nn
from .positional_encoder import PositionalEncoder


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

        self.prenet = nn.Linear(embeddings.size(1), d_model)
        self.pos_encoder = PositionalEncoder(d_model, dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, dim_feedforward=dim_feedforward, nhead=nhead, dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=num_layers,
        )
        self.pred_layer = nn.Sequential(
            nn.Linear(d_model, num_classes),
            # nn.Sigmoid(),
            # nn.Softmax(dim = 1),
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
