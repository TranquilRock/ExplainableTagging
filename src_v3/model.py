import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    

class SeqtoSeqModel(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        d_model: int,
        dim_feedforward: int,
        nhead: int,
        num_layers: int,
        dropout: float,
        num_classes: int,
    ) -> None:

        super(SeqtoSeqModel, self).__init__()
        
        self.embed = nn.Embedding.from_pretrained(embeddings, freeze=False)
        
        num_embeddings, embedding_dim = embeddings.size()

        self.prenet = nn.Linear(embedding_dim, d_model)

        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        self.encoder_layer = nn.TransformerEncoderLayer(
	    d_model=d_model, dim_feedforward=dim_feedforward, nhead=nhead, dropout=dropout
	)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)        

        self.pred_layer = nn.Sequential(
            nn.Linear(d_model, num_classes),
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
