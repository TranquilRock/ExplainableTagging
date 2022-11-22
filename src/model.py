from transformers import LongformerModel
import torch
import torch.nn as nn


class RelationalModel(nn.Module):
    def __init__(self, pretrained_name, num_classes=2, **kwargs):
        super(RelationalModel, self).__init__()
        self.upstream = LongformerModel.from_pretrained(pretrained_name)
        input_dim = self.upstream.config.hidden_size
        self.linear = nn.Linear(input_dim, num_classes)
        self.last = nn.Sigmoid()

    def forward(self, input_ids):

        upstream_outputs = self.upstream(input_ids)
        cls_token_states = upstream_outputs.last_hidden_state[:, 0, :]

        predicted = self.linear(cls_token_states)
        predicted = self.last(predicted)

        return predicted
