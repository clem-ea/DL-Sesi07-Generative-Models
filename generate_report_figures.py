from config import SEED, DEVICE, LATENT_DIM, set_seed

import os
import torch
import matplotlib.pyplot as plt

from dataset import get_dataloaders
from models.vanilla_ae import VanillaAutoencoder
from models.vae import VAE
from models.vanilla_gan import VanillaGenerator
from models.dcgan import DCGANGenerator
from models.ddpm import SimpleUNet, DDPM


def main():
    set_seed(SEED)
    os.makedirs("results", exist_ok=True)
    _, _, test_loader = get_dataloaders(normalize_to_neg_one=False)

    vae = VAE(latent_dim=LATENT_DIM).to(DEVICE)
    vae.load_state_dict(torch.load("checkpoints/vae.pth", map_location=DEVICE))
    vae.eval()

    vanilla_ae = VanillaAutoencoder(latent_dim=LATENT_DIM).to(DEVICE)
    vanilla_ae.load_state_dict(torch.load("checkpoints/vanilla_ae.pth", map_location=DEVICE))
    vanilla_ae.eval()

    v_gen = VanillaGenerator(z_dim=100).to(DEVICE)
    v_gen.load_state_dict(torch.load("checkpoints/vanilla_gan_gen.pth", map_location=DEVICE))
    v_gen.eval()

    dc_gen = DCGANGenerator(z_dim=100).to(DEVICE)
    dc_gen.load_state_dict(torch.load("checkpoints/dcgan_gen.pth", map_location=DEVICE))
    dc_gen.eval()

    unet = SimpleUNet(in_channels=1, base_ch=32, time_emb_dim=128).to(DEVICE)
    ddpm = DDPM(unet, timesteps=200).to(DEVICE)
    ddpm.load_state_dict(torch.load("checkpoints/ddpm.pth", map_location=DEVICE))
    ddpm.eval()

    print("Membuat Figur 2 (Perbandingan Sintesis 4 Model Generatif)...")
    with torch.no_grad():
        z_vae = torch.randn(8, LATENT_DIM, device=DEVICE)
        gen_vae = vae.decode(z_vae).cpu()

        z_gan = torch.randn(8, 100, device=DEVICE)
        gen_vgan = ((v_gen(z_gan) + 1.0) / 2.0).clamp(0.0, 1.0).cpu()
        gen_dcgan = ((dc_gen(z_gan) + 1.0) / 2.0).clamp(0.0, 1.0).cpu()

        gen_ddpm = ((ddpm.sample(8, DEVICE) + 1.0) / 2.0).clamp(0.0, 1.0).cpu()

    fig2_rows = [
        ("VAE (Prior N(0, I))", gen_vae),
        ("Vanilla GAN (MLP)", gen_vgan),
        ("DCGAN (Konvolusional)", gen_dcgan),
        ("DDPM (Denoising T=200)", gen_ddpm),
    ]
    fig, axes = plt.subplots(4, 8, figsize=(14, 7.5))
    for r, (label, imgs) in enumerate(fig2_rows):
        for c in range(8):
            axes[r, c].imshow(imgs[c].squeeze(), cmap="gray", vmin=0, vmax=1)
            axes[r, c].axis("off")
            if c == 0:
                axes[r, c].set_title(label, fontsize=10, loc="left", fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/figur_2_generasi_4model.png", dpi=300)
    plt.close()

    print("Membuat Figur 4 (Interpolasi Ruang Laten: Vanilla AE vs VAE)...")
    sample_x, sample_y = next(iter(test_loader))
    img1 = sample_x[0:1].to(DEVICE)
    idx_diff = (sample_y != sample_y[0]).nonzero(as_tuple=True)[0][0].item()
    img2 = sample_x[idx_diff:idx_diff + 1].to(DEVICE)

    alphas = torch.linspace(0.0, 1.0, steps=10, device=DEVICE)
    with torch.no_grad():
        z1_ae, z2_ae = vanilla_ae.encode(img1), vanilla_ae.encode(img2)
        z1_vae, _ = vae.encode(img1)
        z2_vae, _ = vae.encode(img2)

        interp_ae = [vanilla_ae.decode((1 - a) * z1_ae + a * z2_ae).cpu() for a in alphas]
        interp_vae = [vae.decode((1 - a) * z1_vae + a * z2_vae).cpu() for a in alphas]

    fig, axes = plt.subplots(2, 10, figsize=(16, 3.8))
    for i, a in enumerate(alphas):
        axes[0, i].imshow(interp_ae[i].squeeze(), cmap="gray", vmin=0, vmax=1)
        axes[0, i].axis("off")
        axes[0, i].set_title(f"α={a.item():.1f}", fontsize=8)

        axes[1, i].imshow(interp_vae[i].squeeze(), cmap="gray", vmin=0, vmax=1)
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Vanilla AE", fontsize=10, fontweight="bold")
    axes[1, 0].set_ylabel("VAE", fontsize=10, fontweight="bold")
    axes[0, 0].text(-4, 14, "Vanilla AE\n(Latent Voids)", fontsize=9, fontweight="bold", va="center", ha="right")
    axes[1, 0].text(-4, 14, "VAE\n(Kontinu)", fontsize=9, fontweight="bold", va="center", ha="right")
    plt.tight_layout()
    plt.savefig("results/figur_4_interpolasi_laten.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("Membuat Figur 5 (Kurva Stabilitas Training)...")
    ae_hist = torch.load("results/ae_loss_history.pt")
    gan_hist = torch.load("results/gan_loss_history.pt")
    ddpm_hist = torch.load("results/ddpm_loss_history.pt")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].plot(range(1, len(ae_hist["Vanilla AE"]) + 1), ae_hist["Vanilla AE"], marker="o", label="Vanilla AE (MSE)")
    axes[0].plot(range(1, len(ae_hist["Sparse AE"]) + 1), ae_hist["Sparse AE"], marker="s", label="Sparse AE (MSE)")
    axes[0].set_title("Stabilitas Autoencoder (Konvergen Monoton)", fontsize=10, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Reconstruction MSE Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(range(1, len(gan_hist["Vanilla GAN"]["G_loss"]) + 1), gan_hist["Vanilla GAN"]["G_loss"], linestyle="--", label="Vanilla GAN - G Loss")
    axes[1].plot(range(1, len(gan_hist["Vanilla GAN"]["D_loss"]) + 1), gan_hist["Vanilla GAN"]["D_loss"], linestyle="--", label="Vanilla GAN - D Loss")
    axes[1].plot(range(1, len(gan_hist["DCGAN"]["G_loss"]) + 1), gan_hist["DCGAN"]["G_loss"], marker="o", markersize=4, label="DCGAN - G Loss")
    axes[1].plot(range(1, len(gan_hist["DCGAN"]["D_loss"]) + 1), gan_hist["DCGAN"]["D_loss"], marker="s", markersize=4, label="DCGAN - D Loss")
    axes[1].set_title("Dinamika Adversarial GAN vs DCGAN", fontsize=10, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("BCE Adversarial Loss")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    axes[2].plot(range(1, len(ddpm_hist["DDPM"]) + 1), ddpm_hist["DDPM"], color="purple", marker="^", label="DDPM (Noise MSE)")
    axes[2].set_title("Stabilitas DDPM (Denoising Objective)", fontsize=10, fontweight="bold")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Noise Prediction MSE")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("results/figur_5_stabilitas_training.png", dpi=300)
    plt.close()

    print("\nSelesai! Semua figur laporan (Figur 1 s/d 5) kini lengkap di folder results/.")


if __name__ == "__main__":
    main()