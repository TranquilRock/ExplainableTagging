from transformers import RobertaModel
import torch
import torch.nn as nn
import torch.nn.functional as F

    
class RelationalModel(nn.Module):
    def __init__(self, pretrained_name, num_classes=2, **kwargs):
        super(RelationalModel, self).__init__()

        self.upstream_a = RobertaModel.from_pretrained(pretrained_name)
        self.upstream_b = RobertaModel.from_pretrained(pretrained_name)

        self.embeddings = nn.Embedding(2, 5)
        
        input_dim = self.upstream_a.config.hidden_size + self.upstream_b.config.hidden_size + 5
        self.layer_num = 13
        self.normalize = False
        
        self.weights_a = nn.Parameter(torch.zeros(self.layer_num))
        self.weights_b = nn.Parameter(torch.zeros(self.layer_num))
        
        self.linear = nn.Linear(input_dim, num_classes)
        self.last = nn.Sigmoid()

    def _weighted_sum(self, feature, weights):
        assert self.layer_num == len(feature)
        stacked_feature = torch.stack(feature, dim=0)

        if self.normalize:
            stacked_feature = F.layer_norm(
                stacked_feature, (stacked_feature.shape[-1],)
            )

        _, *origin_shape = stacked_feature.shape
        stacked_feature = stacked_feature.view(self.layer_num, -1)
        norm_weights = F.softmax(weights, dim=-1)
        weighted_feature = (norm_weights.unsqueeze(-1) * stacked_feature).sum(dim=0)
        weighted_feature = weighted_feature.view(*origin_shape)

        return weighted_feature
        
    def forward(self, features_a, features_b, s):        
        features_a = self.upstream_a(features_a, output_hidden_states=True)
        features_a = features_a.hidden_states
        features_a = self._weighted_sum([f for f in features_a], self.weights_a)
        features_a = torch.mean(features_a, dim=1)
        
        features_b = self.upstream_b(features_b, output_hidden_states=True)
        features_b = features_b.hidden_states
        features_b = self._weighted_sum([f for f in features_b], self.weights_b)
        features_b = torch.mean(features_b, dim=1)

        features_s = self.embeddings(s)

        features = torch.cat((features_a, features_b, features_s), dim=-1)
        predicted = self.linear(features)
        predicted = self.last(predicted)

        return predicted
