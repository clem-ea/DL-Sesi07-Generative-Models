from config import SEED, DEVICE, set_seed

import os
import time
import torch
import matplotlib.pyplot as plt
from torchmetrics.image.fid import FrechetInceptionDistance

from dataset import get_dataloaders
from models.ddpm import SimpleUNet, DDPM


def measure_ddpm_latency_ms(ddpm: DDPM, runs: int = 3) -> float:
    _ = ddpm.sample(64, DEVICE)
    start = time.perf_counter()
    for _ in range(runs):
        _ = ddpm.sample(64, DEVICE)
    end = time.perf_counter()
    return ((end - start) / runs) * 1000.0


def compute_ddpm_fid(ddpm: DDPM, test_loader, num_eval_samples: int = 2048) -> float:
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(DEVICE)

    real_count = 0
    for x, _ in test_loader:
        real_01 = ((x.to(DEVICE) + 1.0) / 2.0).clamp(0.0, 1.0).repeat(1, 3, 1, 1)
        fid.update(real_01, real=True)
        real_count += x.size(0)
        if real_count >= num_eval_samples:
            break

    generated = 0
    batch_size = 128
    while generated < num_eval_samples:
        cur_bs = min(batch_size, num_eval_samples - generated)
        fake_01 = ((ddpm.sample(cur_bs, DEVICE) + 1.0) / 2.0).clamp(0.0, 1.0).repeat(1, 3, 1, 1)
        fid.update(fake_01, real=False)
        generated += cur_bs

    return fid.compute().item()


def main():
    set_seed(SEED)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    train_loader, _, test_loader = get_dataloaders(normalize_to_neg_one=True)
    epochs = 12
    timesteps = 200

    print("=== Melatih Denoising Diffusion Probabilistic Model (DDPM) ===")
    unet = SimpleUNet(in_channels=1, base_ch=32, time_emb_dim=128).to(DEVICE)
    ddpm = DDPM(unet, timesteps=timesteps).to(DEVICE)
    optimizer = torch.optim.Adam(ddpm.parameters(), lr=1e-3)

    ddpm_losses = []
    t0 = time.perf_counter()

    for ep in range(1, epochs + 1):
        ddpm.train()
        ep_loss = 0.0
        for x, _ in train_loader:
            x = x.to(DEVICE)
            optimizer.zero_grad()
            loss = ddpm.forward_loss(x)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()

        avg_loss = ep_loss / len(train_loader)
        ddpm_losses.append(avg_loss)
        print(f"[DDPM] Epoch {ep:02d}/{epochs} | Noise MSE Loss: {avg_loss:.5f}")

    time_per_epoch = (time.perf_counter() - t0) / epochs
    torch.save(ddpm.state_dict(), "checkpoints/ddpm.pth")
    torch.save({"DDPM": ddpm_losses}, "results/ddpm_loss_history.pt")

    print("\n=== EVALUASI KUANTITATIF DDPM (BARIS KE-6 TABEL BAGIAN IV) ===")
    ddpm_lat = measure_ddpm_latency_ms(ddpm, runs=3)
    print("Menghitung FID Score DDPM (proses sampling 200 langkah difusi)...")
    ddpm_fid = compute_ddpm_fid(ddpm, test_loader, num_eval_samples=2048)

    print(f"\n6. DDPM | Time/Ep: {time_per_epoch:.2f}s | Latency(64): {ddpm_lat:.2f}ms | MSE: N/A | PSNR: N/A | FID: {ddpm_fid:.2f}")

    samples = ((ddpm.sample(16, DEVICE) + 1.0) / 2.0).clamp(0.0, 1.0).cpu()
    fig, axes = plt.subplots(2, 8, figsize=(14, 3.8))
    for idx in range(16):
        r, c = divmod(idx, 8)
        axes[r, c].imshow(samples[idx].squeeze(), cmap="gray", vmin=0, vmax=1)
        axes[r, c].axis("off")
    plt.suptitle("Sampel Sintesis DDPM (T=200 Langkah Denoising)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/ddpm_samples.png", dpi=300)
    print("Visualisasi sampel DDPM tersimpan di: results/ddpm_samples.png")


if __name__ == "__main__":
    main()