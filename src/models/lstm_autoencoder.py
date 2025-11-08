import torch
import torch.nn as nn

class LSTMAE(nn.Module):
    def __init__(self, d_in, d_hidden=128, d_latent=64, num_layers=1):
        super().__init__()
        self.encoder = nn.LSTM(d_in, d_hidden, num_layers=num_layers, batch_first=True)
        self.to_latent = nn.Linear(d_hidden, d_latent)
        self.from_latent = nn.Linear(d_latent, d_hidden)
        self.decoder = nn.LSTM(d_hidden, d_in, num_layers=num_layers, batch_first=True)

    def forward(self, x):
        # x: (B,L,D)
        h,_ = self.encoder(x)
        h_last = h[:,-1,:]
        z = self.to_latent(h_last)
        h0 = self.from_latent(z).unsqueeze(1).repeat(1, x.size(1), 1)
        x_hat,_ = self.decoder(h0)
        return x_hat
