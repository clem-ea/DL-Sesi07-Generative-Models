from config import SEED, DEVICE, set_seed

import os
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchmetrics.image.fid import FrechetInceptionDistance

from dataset import get_dataloaders
from models.vanilla_gan import VanillaGenerator, VanillaDiscriminator
from models.dcgan import DCGANGenerator, DCGANDiscriminator, weights_init_dcgan


def measure_latency_ms(generator, z_dim: int = 100, runs: int = 50) -> float:
    generator.eval()
    z_sample = torch.randn(64, z_dim, device=DEVICE)
    with torch.no_grad():
        for _ in range(10):
            _ = generator(z_sample)
        start = time.perf_counter()
        for _ in range(runs):
            _ = generator(z_sample)
        end = time.perf_counter()
    return ((end - start) / runs) * 1000.0


def compute_gan_fid(generator, test_loader, z_dim: int = 100) -> float:
    generator.eval()
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(DEVICE)

    with torch.no_grad():
        for x, _ in test_loader:
            real_01 = ((x.to(DEVICE) + 1.0) / 2.0).clamp(0.0, 1.0).repeat(1, 3, 1, 1)
            fid.update(real_01, real=True)

            z = torch.randn(x.size(0), z_dim, device=DEVICE)
            fake_01 = ((generator(z) + 1.0) / 2.0).clamp(0.0, 1.0).repeat(1, 3, 1, 1)
            fid.update(fake_01, real=False)

    return fid.compute().item()


def train_gan_loop(name: str, G, D, train_loader, epochs: int, z_dim: int, lr: float, betas: tuple):
    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=betas)
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=betas)
    criterion = nn.BCEWithLogitsLoss()

    g_losses, d_losses = [], []
    t0 = time.perf_counter()

    for ep in range(1, epochs + 1):
        G.train()
        D.train()
        ep_g_loss, ep_d_loss = 0.0, 0.0

        for real_imgs, _ in train_loader:
            real_imgs = real_imgs.to(DEVICE)
            bs = real_imgs.size(0)
            real_labels = torch.ones(bs, 1, device=DEVICE)
            fake_labels = torch.zeros(bs, 1, device=DEVICE)

            opt_D.zero_grad()
            loss_real = criterion(D(real_imgs), real_labels)

            z = torch.randn(bs, z_dim, device=DEVICE)
            fake_imgs = G(z)
            loss_fake = criterion(D(fake_imgs.detach()), fake_labels)

            loss_D = loss_real + loss_fake
            loss_D.backward()
            opt_D.step()

            opt_G.zero_grad()
            loss_G = criterion(D(fake_imgs), real_labels)
            loss_G.backward()
            opt_G.step()

            ep_d_loss += loss_D.item()
            ep_g_loss += loss_G.item()

        avg_d = ep_d_loss / len(train_loader)
        avg_g = ep_g_loss / len(train_loader)
        d_losses.append(avg_d)
        g_losses.append(avg_g)
        print(f"[{name}] Epoch {ep:02d}/{epochs} | D Loss: {avg_d:.4f} | G Loss: {avg_g:.4f}")

    time_per_epoch = (time.perf_counter() - t0) / epochs
    return time_per_epoch, {"G_loss": g_losses, "D_loss": d_losses}


def main():
    set_seed(SEED)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    train_loader, _, test_loader = get_dataloaders(normalize_to_neg_one=True)
    z_dim = 100
    gan_history = {}

    print("=== 1. Melatih Vanilla GAN (MLP Adversarial) ===")
    v_gen = VanillaGenerator(z_dim=z_dim).to(DEVICE)
    v_disc = VanillaDiscriminator().to(DEVICE)
    time_vgan, gan_history["Vanilla GAN"] = train_gan_loop(
        "Vanilla GAN", v_gen, v_disc, train_loader,
        epochs=20, z_dim=z_dim, lr=2e-4, betas=(0.5, 0.999)
    )
    torch.save(v_gen.state_dict(), "checkpoints/vanilla_gan_gen.pth")

    print("\n=== 2. Melatih Deep Convolutional GAN (DCGAN) ===")
    dc_gen = DCGANGenerator(z_dim=z_dim).to(DEVICE)
    dc_disc = DCGANDiscriminator().to(DEVICE)
    dc_gen.apply(weights_init_dcgan)
    dc_disc.apply(weights_init_dcgan)

    time_dcgan, gan_history["DCGAN"] = train_gan_loop(
        "DCGAN", dc_gen, dc_disc, train_loader,
        epochs=15, z_dim=z_dim, lr=2e-4, betas=(0.5, 0.999)
    )
    torch.save(dc_gen.state_dict(), "checkpoints/dcgan_gen.pth")

    torch.save(gan_history, "results/gan_loss_history.pt")

    print("\n=== EVALUASI KUANTITATIF GAN & DCGAN (TABEL BAGIAN IV) ===")
    vgan_lat = measure_latency_ms(v_gen, z_dim=z_dim)
    print("Menghitung FID Score Vanilla GAN...")
    vgan_fid = compute_gan_fid(v_gen, test_loader, z_dim=z_dim)

    dcgan_lat = measure_latency_ms(dc_gen, z_dim=z_dim)
    print("Menghitung FID Score DCGAN...")
    dcgan_fid = compute_gan_fid(dc_gen, test_loader, z_dim=z_dim)

    print(f"\n4. Vanilla GAN | Time/Ep: {time_vgan:.2f}s | Latency(64): {vgan_lat:.2f}ms | MSE: N/A | PSNR: N/A | FID: {vgan_fid:.2f}")
    print(f"5. DCGAN       | Time/Ep: {time_dcgan:.2f}s | Latency(64): {dcgan_lat:.2f}ms | MSE: N/A | PSNR: N/A | FID: {dcgan_fid:.2f}")

    fixed_z = torch.randn(16, z_dim, device=DEVICE)
    v_gen.eval()
    dc_gen.eval()
    with torch.no_grad():
        v_samples = ((v_gen(fixed_z) + 1.0) / 2.0).clamp(0.0, 1.0).cpu()
        dc_samples = ((dc_gen(fixed_z) + 1.0) / 2.0).clamp(0.0, 1.0).cpu()

    fig, axes = plt.subplots(2, 16, figsize=(18, 3.2))
    for i in range(16):
        axes[0, i].imshow(v_samples[i].squeeze(), cmap="gray", vmin=0, vmax=1)
        axes[0, i].axis("off")
        axes[1, i].imshow(dc_samples[i].squeeze(), cmap="gray", vmin=0, vmax=1)
        axes[1, i].axis("off")
    axes[0, 0].set_title("Vanilla GAN (Rentan Mode Collapse)", fontsize=10, loc="left", fontweight="bold")
    axes[1, 0].set_title("DCGAN (Konvolusional)", fontsize=10, loc="left", fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/gan_vs_dcgan_samples.png", dpi=300)
    print("\nVisualisasi sampel GAN tersimpan di: results/gan_vs_dcgan_samples.png")


if __name__ == "__main__":
    main()