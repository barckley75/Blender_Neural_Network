"""PyTorch training backend for the Blender Neural Network addon.

This module is written so `import neural_network.nn_training` does not fail
when PyTorch is missing — call `has_torch()` to check before instantiating
trainers. The Blender operator uses this to show a friendly error.
"""

from __future__ import annotations

import os
from typing import Optional

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset, random_split

    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    nn = None  # type: ignore
    DataLoader = None  # type: ignore
    TensorDataset = None  # type: ignore
    random_split = None  # type: ignore
    _TORCH_AVAILABLE = False


def has_torch() -> bool:
    return _TORCH_AVAILABLE


def _pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def extract_flat_weights(model: "nn.Module"):
    """Flatten a feed-forward model's weights in the order the GN tree
    iterates connection curves: for each Linear layer in sequence, for each
    source neuron i in 0..M-1, for each target neuron j in 0..N-1 → W[j, i].

    Returns (signed_norm, max_abs). signed_norm is a numpy float32 array
    with values in [-1, 1]; max_abs is the original divisor (useful for
    UI display). If no Linear layers are found, returns an empty array.
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not installed")

    import numpy as np

    parts: list[np.ndarray] = []
    for layer in model:
        if isinstance(layer, nn.Linear):
            w = layer.weight.detach().cpu().numpy()  # shape (out, in)
            parts.append(w.T.reshape(-1).astype(np.float32, copy=False))

    if not parts:
        return np.zeros(0, dtype=np.float32), 0.0

    flat = np.concatenate(parts)
    max_abs = float(np.max(np.abs(flat))) if flat.size else 0.0
    if max_abs <= 0.0:
        return flat.astype(np.float32), 0.0
    return (flat / max_abs).astype(np.float32), max_abs


def build_model(layer_sizes: list[int]) -> "nn.Module":
    """Build a feed-forward MLP from a list of layer sizes.

    A list like [784, 100, 10] yields Linear(784,100) → ReLU → Linear(100,10).
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not installed")
    if len(layer_sizes) < 2:
        raise ValueError("layer_sizes must have at least input and output sizes")

    layers: list[nn.Module] = []
    for i in range(len(layer_sizes) - 1):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        if i < len(layer_sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def _load_dataset(path: str) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Load (X, y) tensors from a dataset file. Supports .npz, .csv, .pt."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        import numpy as np
        data = np.load(path)
        if "X" not in data.files or "y" not in data.files:
            raise ValueError(".npz must contain arrays 'X' and 'y'")
        x = torch.tensor(data["X"], dtype=torch.float32)
        y = torch.tensor(data["y"], dtype=torch.long)
        return x, y

    if ext == ".csv":
        import numpy as np
        arr = np.loadtxt(path, delimiter=",", dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError("CSV must be 2D with at least 2 columns; last column is label")
        x = torch.tensor(arr[:, :-1], dtype=torch.float32)
        y = torch.tensor(arr[:, -1], dtype=torch.long)
        return x, y

    if ext == ".pt":
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict) and "X" in obj and "y" in obj:
            return obj["X"].float(), obj["y"].long()
        if isinstance(obj, (tuple, list)) and len(obj) == 2:
            return obj[0].float(), obj[1].long()
        raise ValueError(".pt file must contain {'X':..., 'y':...} or (X, y) tuple")

    raise ValueError(f"Unsupported dataset extension: {ext}")


class EpochTrainer:
    """Stateful trainer designed to be driven one epoch at a time.

    The Blender operator wraps this in a modal timer so the UI stays
    responsive; each `step()` call runs one training epoch plus validation,
    returning `(epoch, loss, accuracy)` or `None` when training is done.
    """

    def __init__(
        self,
        dataset_path: str,
        layer_sizes: list[int],
        epochs: int,
        minibatch: int,
        lr: float,
    ):
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not installed")

        self.epochs = epochs
        self.device = _pick_device()

        x, y = _load_dataset(dataset_path)
        if x.shape[1] != layer_sizes[0]:
            raise ValueError(
                f"Dataset has {x.shape[1]} features but input layer is {layer_sizes[0]}. "
                "Adjust Input Size to match."
            )
        n_classes = int(y.max().item()) + 1
        if n_classes > layer_sizes[-1]:
            raise ValueError(
                f"Dataset has {n_classes} classes but output layer is {layer_sizes[-1]}. "
                "Adjust Output Size."
            )

        dataset = TensorDataset(x, y)
        val_size = max(1, len(dataset) // 5)
        train_size = len(dataset) - val_size
        self.train_ds, self.val_ds = random_split(
            dataset, [train_size, val_size], generator=torch.Generator().manual_seed(0)
        )
        self.train_loader = DataLoader(self.train_ds, batch_size=minibatch, shuffle=True)
        self.val_loader = DataLoader(self.val_ds, batch_size=minibatch)

        self.model = build_model(layer_sizes).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.CrossEntropyLoss()

        self.current_epoch = 0
        self.last_loss = -1.0
        self.last_accuracy = -1.0

    def flat_weights(self):
        return extract_flat_weights(self.model)

    def step(self) -> Optional[tuple[int, float, float]]:
        if self.current_epoch >= self.epochs:
            return None

        self.model.train()
        total_loss = 0.0
        n_batches = 0
        for xb, yb in self.train_loader:
            xb = xb.to(self.device)
            yb = yb.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(xb)
            loss = self.loss_fn(logits, yb)
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1
        avg_loss = total_loss / max(1, n_batches)

        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in self.val_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                preds = self.model(xb).argmax(dim=1)
                correct += int((preds == yb).sum().item())
                total += int(yb.shape[0])
        acc = correct / max(1, total)

        self.current_epoch += 1
        self.last_loss = avg_loss
        self.last_accuracy = acc
        return self.current_epoch, avg_loss, acc
