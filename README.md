# Comparative Analysis of Generative Models on Fashion-MNIST

Implementasi dan evaluasi komparatif enam arsitektur Deep Generative Models (**Vanilla AE, Sparse AE, VAE, Vanilla GAN, DCGAN, dan DDPM**) serta pembuktian empiris batasan model klasifikasi diskriminatif (**MLP Classifier**) pada dataset **Fashion-MNIST**.

## Tabel Metrik Komparatif (Test Set Fashion-MNIST)

| Arsitektur Model | Dim Laten | Training Time / Epoch (s) | Latency (ms / batch 64) | Test MSE | Test PSNR (dB) | Test SSIM | FID Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Vanilla Autoencoder** | 64 | 49.06 | 1.02 | 0.00905 | 20.43 | 0.8190 | N/A |
| **Sparse Autoencoder** | 64 / 256 | 55.67 | 0.91 | 0.00912 | 20.40 | 0.8194 | N/A |
| **Variational Autoencoder (VAE)** | 64 | 52.45 | 0.58 | 0.01623 | 17.90 | 0.7284 | 96.81 |
| **Vanilla GAN** | 100 | 51.39 | 0.86 | N/A | N/A | N/A | 151.50 |
| **DCGAN** | 100 | 103.81 | 6.09 | N/A | N/A | N/A | 29.93 |
| **DDPM** | 28x28 | 686.19 | 13048.82 | N/A | N/A | N/A | 133.38 |

## Struktur Repositori
- config.py — Konfigurasi global (SEED = 42, BATCH_SIZE = 128, LATENT_DIM = 64).
- dataset.py — Pemuatan dan pembagian deterministik dataset Fashion-MNIST (80% Train / 10% Val / 10% Test).
- models/ — Definisi arsitektur MLPClassifier, VanillaAutoencoder, SparseAutoencoder, VAE, VanillaGenerator/Discriminator, DCGANGenerator/Discriminator, dan DDPM (U-Net).
- un_exp1_mlp_proof.py — Eksperimen pembuktian batasan model diskriminatif MLP (Test Acc: 88.79%).
- 	rain_autoencoders.py — Pelatihan dan evaluasi kuantitatif Vanilla AE, Sparse AE, dan VAE.
- 	rain_gans.py — Pelatihan dan evaluasi kuantitatif Vanilla GAN dan DCGAN.
- 	rain_ddpm.py — Pelatihan dan evaluasi kuantitatif Denoising Diffusion Probabilistic Model (DDPM).
- generate_report_figures.py — Pembuatan grafik dan grid visualisasi komparatif laporan (esults/).
