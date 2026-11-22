from config import SEED, DEVICE, LATENT_DIM, NUM_CLASSES, set_seed

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from dataset import get_dataloaders
from models.mlp_classifier import MLPClassifier, FrozenLatentDecoder


def train_classifier(model, train_loader, val_loader, epochs: int = 15, lr: float = 1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits, _ = model(x)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        val_acc = 100.0 * correct / total
        print(f"[MLP Classifier] Epoch {epoch:02d}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}%")


def evaluate_accuracy(model, test_loader) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits, _ = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total


def train_frozen_decoder(classifier, decoder, train_loader, epochs: int = 15, lr: float = 1e-3):
    classifier.eval()
    for param in classifier.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        decoder.train()
        total_loss = 0.0
        for x, _ in train_loader:
            x = x.to(DEVICE)
            with torch.no_grad():
                z_mlp = classifier.extract_latent(x)
            optimizer.zero_grad()
            x_recon = decoder(z_mlp)
            loss = criterion(x_recon, x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[Frozen Decoder] Epoch {epoch:02d}/{epochs} | Train MSE: {total_loss/len(train_loader):.5f}")


def main():
    set_seed(SEED)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    train_loader, val_loader, test_loader = get_dataloaders(normalize_to_neg_one=False)

    print("=== TAHAP 1: Pelatihan MLP Classifier ===")
    classifier = MLPClassifier(latent_dim=LATENT_DIM, num_classes=NUM_CLASSES).to(DEVICE)
    train_classifier(classifier, train_loader, val_loader, epochs=15)
    test_acc = evaluate_accuracy(classifier, test_loader)
    print(f"--> Akurasi Test Set MLP Classifier: {test_acc:.2f}% (Syarat minimal: 88.00%)\n")
    torch.save(classifier.state_dict(), "checkpoints/mlp_classifier.pth")

    print("=== TAHAP 4: Pelatihan Frozen Latent Decoder dari z_MLP ===")
    decoder = FrozenLatentDecoder(latent_dim=LATENT_DIM).to(DEVICE)
    train_frozen_decoder(classifier, decoder, train_loader, epochs=15)
    torch.save(decoder.state_dict(), "checkpoints/frozen_mlp_decoder.pth")

    classifier.eval()
    decoder.eval()
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)

    total_mse = 0.0
    all_z_mlp = []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(DEVICE)
            z_mlp = classifier.extract_latent(x)
            x_recon = decoder(z_mlp)
            total_mse += F.mse_loss(x_recon, x, reduction="sum").item()
            psnr_metric.update(x_recon, x)
            ssim_metric.update(x_recon, x)
            all_z_mlp.append(z_mlp)

    num_pixels = len(test_loader.dataset) * 28 * 28
    test_mse = total_mse / num_pixels
    test_psnr = psnr_metric.compute().item()
    test_ssim = ssim_metric.compute().item()

    print("\n--> Hasil Evaluasi Frozen Latent Decoder (z_MLP) pada Test Set:")
    print(f"    Test MSE : {test_mse:.5f}")
    print(f"    Test PSNR: {test_psnr:.2f} dB")
    print(f"    Test SSIM: {test_ssim:.4f}\n")

    sample_x, sample_y = next(iter(test_loader))
    sample_x, sample_y = sample_x[:8].to(DEVICE), sample_y[:8].to(DEVICE)

    with torch.no_grad():
        z_sample = classifier.extract_latent(sample_x)
        y_onehot = F.one_hot(sample_y, num_classes=NUM_CLASSES).float()
        x_trans_class = classifier.transpose_from_class(y_onehot)
        x_trans_latent = classifier.transpose_from_latent(z_sample)
        x_frozen_recon = decoder(z_sample)
        all_z_tensor = torch.cat(all_z_mlp, dim=0)
        mu_z = all_z_tensor.mean(dim=0)
        std_z = all_z_tensor.std(dim=0)
        z_rand = mu_z + std_z * torch.randn(8, LATENT_DIM, device=DEVICE)
        x_rand_gen = decoder(z_rand)

    rows = [
        ("1. Ground Truth", sample_x.cpu()),
        ("2a. W^T dari Kelas (y_c)", x_trans_class.cpu()),
        ("2b. W^T dari Laten (z_MLP)", x_trans_latent.cpu()),
        ("3. Frozen Decoder (z_MLP)", x_frozen_recon.cpu()),
        ("4. Random Sampling z_MLP", x_rand_gen.cpu()),
    ]
    fig, axes = plt.subplots(len(rows), 8, figsize=(14, 9))
    for r_idx, (title, imgs) in enumerate(rows):
        for c_idx in range(8):
            axes[r_idx, c_idx].imshow(imgs[c_idx].squeeze(), cmap="gray", vmin=0, vmax=1)
            axes[r_idx, c_idx].axis("off")
            if c_idx == 0:
                axes[r_idx, c_idx].set_title(title, fontsize=10, loc="left", fontweight="bold")

    plt.tight_layout()
    plt.savefig("results/exp1_mlp_proof.png", dpi=300)
    print("Visualisasi tersimpan di: results/exp1_mlp_proof.png")


if __name__ == "__main__":
    main()