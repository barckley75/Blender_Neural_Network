"""Tests the PyTorch training backend. Does not require Blender.

Run with:
    python -m pytest tests/test_training.py

Or standalone:
    python tests/test_training.py
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from neural_network import nn_training


def _make_synthetic_dataset(tmpdir: str, n: int = 64, d: int = 4) -> str:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((n, d)).astype(np.float32)
    y = (x.sum(axis=1) > 0).astype(np.int64)
    path = os.path.join(tmpdir, "synth.npz")
    np.savez(path, X=x, y=y)
    return path


def test_has_torch():
    assert nn_training.has_torch(), "PyTorch must be installed to run training tests"


def test_build_model():
    model = nn_training.build_model([4, 8, 2])
    import torch
    x = torch.randn(3, 4)
    y = model(x)
    assert y.shape == (3, 2)


def test_trainer_runs_and_loss_decreases():
    with tempfile.TemporaryDirectory() as tmp:
        path = _make_synthetic_dataset(tmp)
        trainer = nn_training.EpochTrainer(
            dataset_path=path,
            layer_sizes=[4, 8, 2],
            epochs=10,
            minibatch=16,
            lr=0.01,
        )
        losses = []
        while True:
            result = trainer.step()
            if result is None:
                break
            _, loss, _ = result
            losses.append(loss)

        assert len(losses) == 10
        assert losses[-1] < losses[0], f"Loss did not decrease: {losses}"


def test_trainer_rejects_mismatched_input_size():
    import pytest
    with tempfile.TemporaryDirectory() as tmp:
        path = _make_synthetic_dataset(tmp, d=4)
        with pytest.raises(ValueError, match="features"):
            nn_training.EpochTrainer(
                dataset_path=path,
                layer_sizes=[7, 2],  # 7 != 4 features
                epochs=1,
                minibatch=8,
                lr=0.01,
            )


if __name__ == "__main__":
    test_has_torch()
    test_build_model()
    test_trainer_runs_and_loss_decreases()
    print("OK: all training tests passed")
