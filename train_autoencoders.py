from config import SEED, DEVICE, LATENT_DIM, set_seed

import os
import time
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance

from dataset import get_dataloaders
from models.vanilla_ae import VanillaAutoencoder
from models.sparse_ae import SparseAutoencoder
from models.vae import VAE
from models.mlp_classifier import MLPClassifier, FrozenLatentDecoder


def measure_latency_ms(model_fn, sample_input, runs: int = 50) -> float:
    with torch.no_grad():
        for _ in range(10):  # Warm-up
            _ = model_fn(sample_input)
        start = time.perf_counter()
        for _ in range(runs):
            _ = model_fn(sample_input)
        end = time.perf_counter()
    return ((end - start) / runs) * 1000.0


def evaluate_reconstruction(model, test_loader, is_vae: bool = False, is_sparse: bool = False):
    model.eval()
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
    total_mse = 0.0

    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(DEVICE)
            if is_vae:
                x_recon, _, _ = model(x)
            elif is_sparse:
                x_recon, _, _ = model(x)
            else:
                x_recon, _ = model(x)

            total_mse += F.mse_loss(x_recon, x, reduction="sum").item()
            psnr_metric.update(x_recon, x)
            ssim_metric.update(x_recon, x)

    num_pixels = len(test_loader.dataset) * 28 * 28
    return total_mse / num_pixels, psnr_metric.compute().item(), ssim_metric.compute().item()


def compute_vae_fid(vae_model, test_loader) -> float:
    vae_model.eval()
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(DEVICE)

    with torch.no_grad():
        for x, _ in test_loader:
            x_rgb = x.to(DEVICE).repeat(1, 3, 1, 1)
            fid.update(x_rgb, real=True)

            z = torch.randn(x.size(0), LATENT_DIM, device=DEVICE)
            fake_x = vae_model.decode(z).repeat(1, 3, 1, 1)
            fid.update(fake_x, real=False)

    return fid.compute().item()


def main():
    set_seed(SEED)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    train_loader, _, test_loader = get_dataloaders(normalize_to_neg_one=False)
    epochs = 15
    history = {}

    print("=== 1. Melatih Vanilla Autoencoder ===")
    vanilla_ae = VanillaAutoencoder(latent_dim=LATENT_DIM).to(DEVICE)
    opt_vanilla = torch.optim.Adam(vanilla_ae.parameters(), lr=1e-3)
    history["Vanilla AE"] = []
    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        vanilla_ae.train()
        ep_loss = 0.0
        for x, _ in train_loader:
            x = x.to(DEVICE)
            opt_vanilla.zero_grad()
            x_recon, _ = vanilla_ae(x)
            loss = F.mse_loss(x_recon, x)
            loss.backward()
            opt_vanilla.step()
            ep_loss += loss.item()
        avg_loss = ep_loss / len(train_loader)
        history["Vanilla AE"].append(avg_loss)
        print(f"[Vanilla AE] Epoch {ep:02d}/{epochs} | Train MSE: {avg_loss:.5f}")
    time_vanilla = (time.perf_counter() - t0) / epochs
    torch.save(vanilla_ae.state_dict(), "checkpoints/vanilla_ae.pth")

    print("\n=== 2. Melatih Sparse Autoencoder ===")
    sparse_ae = SparseAutoencoder(sparse_dim=256, latent_dim=LATENT_DIM).to(DEVICE)
    opt_sparse = torch.optim.Adam(sparse_ae.parameters(), lr=1e-3)
    lambda_l1 = 1e-4
    history["Sparse AE"] = []
    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        sparse_ae.train()
        ep_loss = 0.0
        for x, _ in train_loader:
            x = x.to(DEVICE)
            opt_sparse.zero_grad()
            x_recon, _, h_sparse = sparse_ae(x)
            recon_loss = F.mse_loss(x_recon, x)
            sparsity_penalty = lambda_l1 * torch.mean(torch.abs(h_sparse))
            loss = recon_loss + sparsity_penalty
            loss.backward()
            opt_sparse.step()
            ep_loss += recon_loss.item()
        avg_loss = ep_loss / len(train_loader)
        history["Sparse AE"].append(avg_loss)
        print(f"[Sparse AE]  Epoch {ep:02d}/{epochs} | Train MSE: {avg_loss:.5f}")
    time_sparse = (time.perf_counter() - t0) / epochs
    torch.save(sparse_ae.state_dict(), "checkpoints/sparse_ae.pth")

    print("\n=== 3. Melatih Variational Autoencoder (VAE) ===")
    vae = VAE(latent_dim=LATENT_DIM).to(DEVICE)
    opt_vae = torch.optim.Adam(vae.parameters(), lr=1e-3)
    history["VAE"] = []
    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        vae.train()
        ep_loss = 0.0
        for x, _ in train_loader:
            x = x.to(DEVICE)
            opt_vae.zero_grad()
            x_recon, mu, logvar = vae(x)
            bce = F.binary_cross_entropy(x_recon, x, reduction="sum") / x.size(0)
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
            loss = bce + kld
            loss.backward()
            opt_vae.step()
            ep_loss += loss.item()
        avg_loss = ep_loss / len(train_loader)
        history["VAE"].append(avg_loss)
        print(f"[VAE]        Epoch {ep:02d}/{epochs} | Train ELBO Loss: {avg_loss:.2f}")
    time_vae = (time.perf_counter() - t0) / epochs
    torch.save(vae.state_dict(), "checkpoints/vae.pth")

    torch.save(history, "results/ae_loss_history.pt")

    print("\n=== EVALUASI KUANTITATIF TEST SET (TABEL BAGIAN IV) ===")
    batch_64 = next(iter(test_loader))[0][:64].to(DEVICE)
    z_64 = torch.randn(64, LATENT_DIM, device=DEVICE)

    v_mse, v_psnr, v_ssim = evaluate_reconstruction(vanilla_ae, test_loader)
    v_lat = measure_latency_ms(lambda inp: vanilla_ae(inp)[0], batch_64)

    s_mse, s_psnr, s_ssim = evaluate_reconstruction(sparse_ae, test_loader, is_sparse=True)
    s_lat = measure_latency_ms(lambda inp: sparse_ae(inp)[0], batch_64)

    vae_mse, vae_psnr, vae_ssim = evaluate_reconstruction(vae, test_loader, is_vae=True)
    vae_lat = measure_latency_ms(lambda z: vae.decode(z), z_64)
    print("Menghitung FID Score VAE (mengunduh Inception-v3 jika pertama kali)...")
    vae_fid = compute_vae_fid(vae, test_loader)

    print(f"\n1. Vanilla AE | Time/Ep: {time_vanilla:.2f}s | Latency(64): {v_lat:.2f}ms | MSE: {v_mse:.5f} | PSNR: {v_psnr:.2f} dB | SSIM: {v_ssim:.4f} | FID: N/A")
    print(f"2. Sparse AE  | Time/Ep: {time_sparse:.2f}s | Latency(64): {s_lat:.2f}ms | MSE: {s_mse:.5f} | PSNR: {s_psnr:.2f} dB | SSIM: {s_ssim:.4f} | FID: N/A")
    print(f"3. VAE        | Time/Ep: {time_vae:.2f}s | Latency(64): {vae_lat:.2f}ms | MSE: {vae_mse:.5f} | PSNR: {vae_psnr:.2f} dB | SSIM: {vae_ssim:.4f} | FID: {vae_fid:.2f}")

    sample_x, sample_y = next(iter(test_loader))
    sample_x, sample_y = sample_x[:8].to(DEVICE), sample_y[:8].to(DEVICE)

    vanilla_ae.eval()
    sparse_ae.eval()
    vae.eval()
    with torch.no_grad():
        rec_vanilla, _ = vanilla_ae(sample_x)
        rec_sparse, _, _ = sparse_ae(sample_x)
        rec_vae, _, _ = vae(sample_x)

    fig1_rows = [
        ("Citra Asli (Ground Truth)", sample_x.cpu()),
        ("Vanilla Autoencoder", rec_vanilla.cpu()),
        ("Sparse Autoencoder", rec_sparse.cpu()),
        ("Variational Autoencoder (VAE)", rec_vae.cpu()),
    ]
    fig, axes = plt.subplots(4, 8, figsize=(14, 7.5))
    for r, (label, imgs) in enumerate(fig1_rows):
        for c in range(8):
            axes[r, c].imshow(imgs[c].squeeze(), cmap="gray", vmin=0, vmax=1)
            axes[r, c].axis("off")
            if c == 0:
                axes[r, c].set_title(label, fontsize=10, loc="left", fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/figur_1_rekonstruksi.png", dpi=300)

    clf = MLPClassifier(latent_dim=LATENT_DIM).to(DEVICE)
    dec_mlp = FrozenLatentDecoder(latent_dim=LATENT_DIM).to(DEVICE)
    clf.load_state_dict(torch.load("checkpoints/mlp_classifier.pth", map_location=DEVICE))
    dec_mlp.load_state_dict(torch.load("checkpoints/frozen_mlp_decoder.pth", map_location=DEVICE))
    clf.eval()
    dec_mlp.eval()

    with torch.no_grad():
        z_mlp = clf.extract_latent(sample_x)
        y_oh = F.one_hot(sample_y, num_classes=10).float()
        wt_class = clf.transpose_from_class(y_oh)
        wt_latent = clf.transpose_from_latent(z_mlp)
        frozen_rec = dec_mlp(z_mlp)
        z_rand = z_mlp.mean(dim=0) + z_mlp.std(dim=0) * torch.randn(8, LATENT_DIM, device=DEVICE)
        mlp_rand = dec_mlp(z_rand)

    fig3_rows = [
        ("Panel 1: Ground Truth", sample_x.cpu()),
        ("Panel 2a: W^T dari Kelas (y_c)", wt_class.cpu()),
        ("Panel 2b: W^T dari Laten (z_MLP)", wt_latent.cpu()),
        ("Panel 3a: Frozen Decoder (z_MLP)", frozen_rec.cpu()),
        ("Panel 3b: Vanilla AE (z_AE)", rec_vanilla.cpu()),
        ("Panel 4: Prior Sampling (z_MLP)", mlp_rand.cpu()),
    ]
    fig, axes = plt.subplots(len(fig3_rows), 8, figsize=(14, 10.5))
    for r, (label, imgs) in enumerate(fig3_rows):
        for c in range(8):
            axes[r, c].imshow(imgs[c].squeeze(), cmap="gray", vmin=0, vmax=1)
            axes[r, c].axis("off")
            if c == 0:
                axes[r, c].set_title(label, fontsize=10, loc="left", fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/figur_3_mlp_proof.png", dpi=300)
    print("\nFigur 1 dan Figur 3 berhasil disimpan di folder results/!")


if __name__ == "__main__":
    main()