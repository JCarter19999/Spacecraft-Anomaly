import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x):
        L = x.size(1)
        return x + self.pe[:L].unsqueeze(0)

class TransformerClassifier(nn.Module):
    def __init__(self, d_in, d_model=128, nhead=8, num_layers=2, dim_ff=256, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, dropout=dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos = PositionalEncoding(d_model)
        self.head = nn.Linear(d_model, 1)
        self.act = nn.Sigmoid()
    def forward(self, x):
        z = self.proj(x)
        z = self.pos(z)
        z = self.enc(z)
        z = z.mean(dim=1)
        return self.act(self.head(z).squeeze(1))
