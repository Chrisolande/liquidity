"""
PyTorch Neural TabMLP Base Learner with cross-validation and GPU acceleration.
"""

from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import HAS_GPU, SEED
from src.metrics import competition_score


class TabMLP(nn.Module):
    """Deep residual tabular multilayer perceptron for liquidity stress representations."""

    def __init__(self, in_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def fit_mlp_runner(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    feature_cols: List[str],
    epochs: int = 25,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Fits 10-fold PyTorch TabMLP with AdamW, BCEWithLogitsLoss, and CosineAnnealingLR.
    """
    device = torch.device("cuda:0" if HAS_GPU and torch.cuda.is_available() else "cpu")
    print(f"Training PyTorch TabMLP on {device}", flush=True)

    X_tr_mat = X_train[feature_cols].to_numpy(dtype=np.float32)
    X_te_mat = X_test[feature_cols].to_numpy(dtype=np.float32)

    # Impute missing with means
    col_means = np.nanmean(X_tr_mat, axis=0)
    col_means = np.nan_to_num(col_means, nan=0.0)
    inds_tr = np.where(np.isnan(X_tr_mat))
    X_tr_mat[inds_tr] = np.take(col_means, inds_tr[1])
    inds_te = np.where(np.isnan(X_te_mat))
    X_te_mat[inds_te] = np.take(col_means, inds_te[1])

    in_dim = X_tr_mat.shape[1]
    n_train = len(X_tr_mat)
    n_test = len(X_te_mat)
    oof_mlp = np.zeros(n_train, dtype=float)
    test_mlp = np.zeros(n_test, dtype=float)
    criterion = nn.BCEWithLogitsLoss()

    X_te_tensor = torch.tensor(X_te_mat, dtype=torch.float32).to(device)

    for fold, (trn_idx, val_idx) in enumerate(folds):
        torch.manual_seed(SEED + fold)
        model = TabMLP(in_dim).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        scaler = StandardScaler()
        x_tr_scaled = scaler.fit_transform(X_tr_mat[trn_idx])
        x_va_scaled = scaler.transform(X_tr_mat[val_idx])
        x_te_scaled = scaler.transform(X_te_mat)

        x_trn_fold = torch.tensor(x_tr_scaled, dtype=torch.float32)
        y_trn_fold = torch.tensor(y_train[trn_idx], dtype=torch.float32)
        x_val_fold = torch.tensor(x_va_scaled, dtype=torch.float32).to(device)
        x_test_fold = torch.tensor(x_te_scaled, dtype=torch.float32).to(device)

        trn_ds = TensorDataset(x_trn_fold, y_trn_fold)
        trn_loader = DataLoader(trn_ds, batch_size=batch_size, shuffle=True, drop_last=False)

        for ep in range(epochs):
            model.train()
            for bx, by in trn_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
            scheduler.step()

        model.eval()
        with torch.no_grad():
            va_logits = model(x_val_fold).cpu().numpy()
            te_logits = model(x_test_fold).cpu().numpy()
            oof_mlp[val_idx] = 1.0 / (1.0 + np.exp(-va_logits))
            test_mlp += (1.0 / (1.0 + np.exp(-te_logits))) / len(folds)

    ll, auc, comp = competition_score(y_train, oof_mlp)
    print(f"TabMLP 10-Fold OOF: LL={ll:.5f}, AUC={auc:.5f}, Comp={comp:.5f}", flush=True)
    return oof_mlp, test_mlp, comp
