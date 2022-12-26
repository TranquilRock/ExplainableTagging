"""TODO"""

import torch
import torch.nn as nn
from .positional_encoder import PositionalEncoder


class QDNet(torch.nn.Module):
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
        super().__init__()

        self.embed = nn.Sequential(
            nn.Embedding.from_pretrained(embeddings, freeze=True),
            nn.Linear(embeddings.size(1), d_model),
        )

        self.encode = PositionalEncoder(d_model, dropout)

        # self.query_tranformer = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(
        #         d_model=d_model,
        #         dim_feedforward=dim_feedforward,
        #         nhead=nhead,
        #         dropout=dropout,
        #     ),
        #     num_layers=num_layers,
        # )

        self.query_tranformer = self.document_tranformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                dim_feedforward=dim_feedforward,
                nhead=nhead,
                dropout=dropout,
            ),
            num_layers=num_layers,
        )

        self.cls = nn.Sequential(
            nn.Linear(d_model, num_classes),
        )

    def forward(
        self, query_tokens: torch.LongTensor, article_tokens: torch.LongTensor
    ) -> torch.LongTensor:
        """TODO"""
        length = article_tokens.size(1)
        qry: torch.Tensor = self.embed(torch.concat(
            (article_tokens, query_tokens), dim=1))
        qry = qry.permute(1, 0, 2)
        qry = self.encode(qry)
        # (length, batch size, d_model).
        qry = self.query_tranformer(qry)
        # Take the first embeding of the sequence
        qry = qry[0, :, :].unsqueeze(0)

        art: torch.Tensor = self.embed(article_tokens)
        # out: (length, batch size, d_model)
        art = art.permute(1, 0, 2)
        art = self.encode(art)
        art = torch.concat((qry, art))
        # Ignore the first output of query
        art = self.document_tranformer(art)[:length]
        return self.cls(art.transpose(0, 1))
