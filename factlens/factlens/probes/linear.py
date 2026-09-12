"""A linear probe trained from scratch with AdamW — no sklearn, no TRL.

The probe is a single linear map from a hidden state to the relation's
label space (the set of objects). Training uses mini-batch SGD (AdamW) with
early stopping on validation macro-F1 and best-state restoration, which is
the standard recipe for reporting probe accuracy without overfitting
inflation (Belinkov, 2022, §4).

Optionally ``bias=False`` trains a direction-only probe (a scaled unit
linear form), which is the flavor used when one cares about the *geometry*
of the decoding direction rather than raw separability.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from .metrics import macro_f1


class LinearProbe(nn.Module):
    """L2-regularized linear classifier over hidden states.

    Args:
        d_model: Input dimensionality of the hidden states.
        n_classes: Size of the relation's label space.
        bias: Learn an explicit bias term.
        lr: AdamW learning rate.
        weight_decay: AdamW weight decay (the L2 regularizer).
    """

    def __init__(
        self,
        d_model: int,
        n_classes: int,
        bias: bool = True,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.fc = nn.Linear(d_model, n_classes, bias=bias)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.loss_fn = nn.CrossEntropyLoss()
        self.best_state: dict | None = None
        self.best_score: float = -1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

    # -- training --------------------------------------------------------

    def fit(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_val: torch.Tensor | None = None,
        y_val: torch.Tensor | None = None,
        epochs: int = 300,
        batch_size: int = 64,
        patience: int = 30,
        verbose: bool = False,
    ) -> dict:
        """Train with early stopping; restores the best weights.

        Returns:
            History dict with per-epoch train loss and validation macro-F1.
        """
        history = {"train_loss": [], "val_f1": []}
        if x_val is None:
            x_val, y_val = x_train, y_train

        n = len(x_train)
        epochs_iter = tqdm(range(epochs), desc="probe", leave=False) if verbose else range(epochs)
        stale = 0
        for epoch in epochs_iter:
            self.train()
            perm = torch.randperm(n)
            losses = []
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                logits = self(x_train[idx])
                loss = self.loss_fn(logits, y_train[idx])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.append(float(loss.item()))
            history["train_loss"].append(float(np.mean(losses)))

            self.eval()
            with torch.no_grad():
                val_logits = self(x_val)
                val_pred = val_logits.argmax(dim=-1).cpu().numpy()
                val_f1 = macro_f1(y_val.cpu().numpy(), val_pred, self.fc.out_features)
            history["val_f1"].append(val_f1)

            if val_f1 > self.best_score + 1e-6:
                self.best_score = val_f1
                self.best_state = copy.deepcopy(self.state_dict())
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break

        if self.best_state is not None:
            self.load_state_dict(self.best_state)
        self.eval()
        return history

    # -- inference --------------------------------------------------------

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> np.ndarray:
        return self(x).argmax(dim=-1).cpu().numpy()

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        return torch.softmax(self(x), dim=-1).cpu().numpy()
