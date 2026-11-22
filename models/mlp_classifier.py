import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int = 784, hidden_dim: int = 256, latent_dim: int = 64, num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, num_classes)

    def extract_latent(self, x: torch.Tensor) -> torch.Tensor:
        x_flat = x.view(x.size(0), -1)
        h1 = F.relu(self.fc1(x_flat))
        z_mlp = F.relu(self.fc2(h1))
        return z_mlp

    def forward(self, x: torch.Tensor):
        z_mlp = self.extract_latent(x)
        logits = self.fc3(z_mlp)
        return logits, z_mlp

    @torch.no_grad()
    def transpose_from_class(self, y_onehot: torch.Tensor) -> torch.Tensor:
        z_hat = F.relu(F.linear(y_onehot, self.fc3.weight.t()))
        h1_hat = F.relu(F.linear(z_hat, self.fc2.weight.t()))
        x_hat = torch.sigmoid(F.linear(h1_hat, self.fc1.weight.t()))
        return x_hat.view(-1, 1, 28, 28)

    @torch.no_grad()
    def transpose_from_latent(self, z_mlp: torch.Tensor) -> torch.Tensor:
        h1_hat = F.relu(F.linear(z_mlp, self.fc2.weight.t()))
        x_hat = torch.sigmoid(F.linear(h1_hat, self.fc1.weight.t()))
        return x_hat.view(-1, 1, 28, 28)


class FrozenLatentDecoder(nn.Module):
    def __init__(self, latent_dim: int = 64, hidden_dim: int = 256, output_dim: int = 784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.net(z)
        return out.view(-1, 1, 28, 28)