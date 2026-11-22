import torch
import torch.nn as nn


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int = 784, sparse_dim: int = 256, latent_dim: int = 64):
        super().__init__()
        self.enc_sparse = nn.Sequential(
            nn.Linear(input_dim, sparse_dim),
            nn.ReLU()
        )
        self.enc_bottleneck = nn.Sequential(
            nn.Linear(sparse_dim, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, sparse_dim),
            nn.ReLU(),
            nn.Linear(sparse_dim, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h_sparse = self.enc_sparse(x.view(x.size(0), -1))
        return self.enc_bottleneck(h_sparse)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z).view(-1, 1, 28, 28)

    def forward(self, x: torch.Tensor):
        h_sparse = self.enc_sparse(x.view(x.size(0), -1))
        z = self.enc_bottleneck(h_sparse)
        x_recon = self.decode(z)
        return x_recon, z, h_sparse