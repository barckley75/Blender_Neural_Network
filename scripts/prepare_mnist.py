"""Download MNIST and save it as datasets/mnist.npz for the addon.

Run with Blender's bundled Python (the one that has torch / torchvision):

    "<blender python>" scripts/prepare_mnist.py

Output: datasets/mnist.npz with arrays X (N, 784) float32 in [0, 1] and
y (N,) int64. Train + test are concatenated (70k samples total).
"""

from __future__ import annotations

import os
import ssl
import sys

import numpy as np

try:
    from torchvision.datasets import MNIST
except ImportError:
    sys.stderr.write(
        "torchvision is required. Install it into Blender's Python:\n"
        '  "<blender python>" -m pip install torchvision\n'
    )
    sys.exit(1)

try:
    import certifi

    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    ssl._create_default_https_context = lambda: ssl.create_default_context(
        cafile=certifi.where()
    )
except ImportError:
    pass


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(REPO_ROOT, "datasets", "_mnist_cache")
OUT_PATH = os.path.join(REPO_ROOT, "datasets", "mnist.npz")


def _to_arrays(ds: MNIST) -> tuple[np.ndarray, np.ndarray]:
    x = ds.data.numpy().astype(np.float32) / 255.0
    x = x.reshape(x.shape[0], -1)
    y = ds.targets.numpy().astype(np.int64)
    return x, y


def main() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"Downloading MNIST into {CACHE_DIR} (skipped if already cached)...")
    train = MNIST(CACHE_DIR, train=True, download=True)
    test = MNIST(CACHE_DIR, train=False, download=True)

    x_train, y_train = _to_arrays(train)
    x_test, y_test = _to_arrays(test)
    x = np.concatenate([x_train, x_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.savez(OUT_PATH, X=x, y=y)
    print(f"Wrote {OUT_PATH}  X={x.shape} {x.dtype}  y={y.shape} {y.dtype}")
    print("Point the Training panel's Dataset Path at this file.")


if __name__ == "__main__":
    main()
