"""CPU-only synthetic check of both trainers, calibration, evaluation and cascade."""

import argparse
import json
from pathlib import Path

import numpy as np

from .evaluate_models import evaluate
from .light_world_model import Pruner
from .prepare_training import FeatureData, prepare
from .run_cascade import cascade
from .train_models import train


def synthetic_data(path, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for episode in range(20):
        for group in range(2):
            state = rng.uniform(-0.5, 0.5, 2).astype(np.float32)
            target = state + rng.choice([-1.0, 1.0], size=2).astype(np.float32) * 0.3
            z = np.tile(np.tile(state, 4), (4, 1))
            goal = np.tile(target, 4)
            for candidate in range(8):
                # Analytic 2-D dynamics; candidate population has good/bad actions.
                center = (target - state) / 0.5 * (1 if candidate < 4 else -1)
                actions = np.clip(center + rng.normal(0, 0.25, (4, 2)), -1, 1).astype(
                    np.float32
                )
                next_state = state + 0.5 * actions.mean(0)
                rows.append(
                    {
                        "z": z,
                        "future": np.tile(np.tile(next_state, 4), (4, 1)),
                        "goal": goal,
                        "state": state,
                        "actions": actions,
                        "success": float(np.linalg.norm(next_state - target) < 0.25),
                        "episode": f"synthetic-{episode}",
                        "branch_group": f"synthetic-{episode}-{group}",
                    }
                )
    meta = {
        "feature_id": "SYNTHETIC_ONLY",
        "demo": True,
        "controller": {"low": [-1.0, -1.0], "high": [1.0, 1.0], "clip": True},
        "outcome_protocol": {"kind": "synthetic_2d_analytic_dynamics"},
    }
    data = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
    np.savez_compressed(path, **data, metadata=np.array(json.dumps(meta)))
    return data


def run(output="training_smoke", epochs=4):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    synthetic_data(root / "synthetic.npz")
    prepare([root / "synthetic.npz"], root / "cache")
    for size, width, depth in [("light", 32, 2), ("medium", 64, 3)]:
        train(
            root / "cache",
            root / f"{size}.pt",
            size=size,
            epochs=epochs,
            batch_size=32,
            width=width,
            depth=depth,
            heads=4,
            lr=1e-3,
            warmup_epochs=1,
            patience=5,
        )
    report = evaluate(
        root / "cache",
        root / "light.pt",
        root / "medium.pt",
        root / "comparison.json",
        keep=2,
        allow_demo=True,
        benchmark_groups=1,
    )
    data = FeatureData(root / "cache")
    ids = data.splits["test"]
    g = data.arrays["branch_group"][ids[0]]
    ix = ids[data.arrays["branch_group"][ids] == g]
    f = {k: np.asarray(data.arrays[k][ix[0]]) for k in ("z", "state", "goal")}
    light, medium = [
        Pruner.load(root / f"{s}.pt", allow_demo=True) for s in ("light", "medium")
    ]
    out = cascade(
        light, medium, f, np.asarray(data.arrays["actions"][ix]), "SYNTHETIC_ONLY", 2, 1
    )
    np.savez_compressed(root / "demo_cascade.npz", **out)
    assert len(out["indices"]) == 1 and len(out["survivor_indices"]) == 2
    assert np.isfinite(out["medium_scores"]).sum() == 2
    for size in ("light", "medium"):
        try:
            Pruner.load(root / f"{size}.pt")
        except ValueError:
            pass
        else:
            raise AssertionError("Demo checkpoint accepted for real inference")
    print(
        "SYNTHETIC ONLY: both models trained, calibrated and evaluated; no robot-quality claim."
    )
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="training_smoke")
    p.add_argument("--epochs", type=int, default=4)
    run(**vars(p.parse_args()))
