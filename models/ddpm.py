import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None].float() * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(8, out_ch)
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(8, out_ch)
        self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.gn1(self.conv1(x)))
        h = h + self.time_mlp(F.silu(t_emb))[:, :, None, None]
        h = F.silu(self.gn2(self.conv2(h)))
        return h + self.shortcut(x)


class SimpleUNet(nn.Module):
    def __init__(self, in_channels: int = 1, base_ch: int = 32, time_emb_dim: int = 128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU()
        )

        self.inc = ResBlock(in_channels, base_ch, time_emb_dim)
        self.down1 = nn.Conv2d(base_ch, base_ch, kernel_size=4, stride=2, padding=1)
        self.enc1 = ResBlock(base_ch, base_ch * 2, time_emb_dim)
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 2, kernel_size=4, stride=2, padding=1)

        self.bot = ResBlock(base_ch * 2, base_ch * 2, time_emb_dim)

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch * 2, kernel_size=4, stride=2, padding=1)
        self.dec1 = ResBlock(base_ch * 4, base_ch, time_emb_dim)
        self.up2 = nn.ConvTranspose2d(base_ch, base_ch, kernel_size=4, stride=2, padding=1)
        self.dec2 = ResBlock(base_ch * 2, base_ch, time_emb_dim)

        self.outc = nn.Conv2d(base_ch, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)
        x1 = self.inc(x, t_emb)
        x2 = self.enc1(self.down1(x1), t_emb)
        x3 = self.bot(self.down2(x2), t_emb)

        u1 = self.up1(x3)
        u1 = self.dec1(torch.cat([u1, x2], dim=1), t_emb)
        u2 = self.up2(u1)
        u2 = self.dec2(torch.cat([u2, x1], dim=1), t_emb)
        return self.outc(u2)


class DDPM(nn.Module):
    def __init__(self, model: nn.Module, timesteps: int = 200, beta_start: float = 1e-4, beta_end: float = 0.02):
        super().__init__()
        self.model = model
        self.timesteps = timesteps

        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)

    def forward_loss(self, x_0: torch.Tensor) -> torch.Tensor:
        bs = x_0.size(0)
        t = torch.randint(0, self.timesteps, (bs,), device=x_0.device).long()
        noise = torch.randn_like(x_0)

        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]

        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
        predicted_noise = self.model(x_t, t)
        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        self.model.eval()
        x = torch.randn(num_samples, 1, 28, 28, device=device)

        for i in reversed(range(self.timesteps)):
            t = torch.full((num_samples,), i, device=device, dtype=torch.long)
            pred_noise = self.model(x, t)

            alpha_t = self.alphas[i]
            alpha_bar_t = self.alphas_cumprod[i]
            beta_t = self.betas[i]

            noise = torch.randn_like(x) if i > 0 else torch.zeros_like(x)
            x = (1.0 / torch.sqrt(alpha_t)) * (
                x - ((1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)) * pred_noise
            ) + torch.sqrt(beta_t) * noise

        return x.clamp(-1.0, 1.0)