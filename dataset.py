import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import random
import numpy as np
import torch
from config import SEED, BATCH_SIZE
import torch
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torchvision import datasets, transforms

def get_dataloaders(data_dir: str = "./data", normalize_to_neg_one: bool = False):
    if normalize_to_neg_one:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

    train_raw = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
    test_raw = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)
    full_dataset = ConcatDataset([train_raw, test_raw])

    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(SEED)
    train_set, val_set, test_set = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    tr, va, te = get_dataloaders()
    print(f"Train batches: {len(tr)} ({len(tr.dataset)} citra)")
    print(f"Val batches  : {len(va)} ({len(va.dataset)} citra)")
    print(f"Test batches : {len(te)} ({len(te.dataset)} citra)")