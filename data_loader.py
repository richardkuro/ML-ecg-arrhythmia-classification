"""
data_loader.py
--------------
Loads the Kaggle ECG Heartbeat dataset (MIT-BIH + PTB).

Expected files (place in ./data/):
  mitbih_train.csv  mitbih_test.csv
  ptbdb_normal.csv  ptbdb_abnormal.csv

Each row = 187 signal values + 1 label column (last column).
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import label_binarize
from collections import Counter


# ── Dataset class ─────────────────────────────────────────────────────────────

class ECGDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        # X shape: (N, 187)  →  model sees (N, 1, 187)
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_mitbih(data_dir: str = "data"):
    """Returns train/test splits for MIT-BIH (5-class arrhythmia)."""
    train_path = os.path.join(data_dir, "mitbih_train.csv")
    test_path  = os.path.join(data_dir, "mitbih_test.csv")

    train_df = pd.read_csv(train_path, header=None)
    test_df  = pd.read_csv(test_path,  header=None)

    X_train, y_train = train_df.iloc[:, :-1].values, train_df.iloc[:, -1].values.astype(int)
    X_test,  y_test  = test_df.iloc[:, :-1].values,  test_df.iloc[:, -1].values.astype(int)

    return X_train, X_test, y_train, y_test


def load_ptbdb(data_dir: str = "data"):
    """Returns train/test splits for PTB-DB (2-class: normal vs MI)."""
    normal_df   = pd.read_csv(os.path.join(data_dir, "ptbdb_normal.csv"),   header=None)
    abnormal_df = pd.read_csv(os.path.join(data_dir, "ptbdb_abnormal.csv"), header=None)

    df = pd.concat([normal_df, abnormal_df], ignore_index=True).sample(frac=1, random_state=42)
    X  = df.iloc[:, :-1].values
    y  = df.iloc[:, -1].values.astype(int)

    split = int(0.8 * len(y))
    return X[:split], X[split:], y[:split], y[split:]


# ── Weighted sampler (handles class imbalance in MIT-BIH) ─────────────────────

def make_weighted_sampler(y: np.ndarray) -> WeightedRandomSampler:
    counts  = Counter(y)
    weights = np.array([1.0 / counts[label] for label in y])
    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )


# ── DataLoader factory ────────────────────────────────────────────────────────

def get_loaders(dataset: str = "mitbih",
                data_dir: str = "data",
                batch_size: int = 256,
                balance: bool = True,
                num_workers: int = 2):
    """
    dataset : 'mitbih' | 'ptbdb'
    balance : oversample minority classes via WeightedRandomSampler (train only)
    """
    if dataset == "mitbih":
        X_train, X_test, y_train, y_test = load_mitbih(data_dir)
        num_classes = 5
    else:
        X_train, X_test, y_train, y_test = load_ptbdb(data_dir)
        num_classes = 2

    train_ds = ECGDataset(X_train, y_train)
    test_ds  = ECGDataset(X_test,  y_test)

    sampler = make_weighted_sampler(y_train) if balance else None
    shuffle = not balance  # mutually exclusive with sampler

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=sampler, shuffle=shuffle,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)

    print(f"[{dataset.upper()}] train={len(train_ds):,}  test={len(test_ds):,}  classes={num_classes}")
    print(f"  Class distribution (train): { {k: int(v) for k, v in sorted(Counter(y_train).items())} }")
    return train_loader, test_loader, num_classes


if __name__ == "__main__":
    tl, vl, nc = get_loaders("mitbih")
    xb, yb = next(iter(tl))
    print(f"Batch X: {xb.shape}  y: {yb.shape}  classes: {nc}")
