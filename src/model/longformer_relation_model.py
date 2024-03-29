"""Models for relation task."""
from transformers import LongformerModel
import torch
import torch.nn as nn


class LongformerRelationModel(nn.Module):
    """Relational model based on Longformer."""

    def __init__(self, pretrained_name, num_classes=2, **kwargs):
        super(LongformerRelationModel, self).__init__()

        self.upstream = LongformerModel.from_pretrained(pretrained_name)

        input_dim = self.upstream.config.hidden_size

        self.downstream = nn.Linear(input_dim, num_classes)

        self.last = nn.Sigmoid()

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        features = self.upstream(tokens, output_hidden_states=True)
        features = features.hidden_states[-1][:, 0, :]

        predicted = self.downstream(features)
        predicted = self.last(predicted)

        return predicted
