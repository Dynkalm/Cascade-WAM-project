"""Compare batched adapter results against native OpenPI with identical noise."""

import argparse

import numpy as np

from .pi05_adapter import Pi05Adapter, load_observation

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--observation", required=True)
    p.add_argument("--checkpoint", default="gs://openpi-assets/checkpoints/pi05_libero")
    p.add_argument("--config", default="pi05_libero")
    p.add_argument("--candidates", type=int, default=1)
    p.add_argument("--vla-batch", type=int, default=1)
    a = p.parse_args()
    if min(a.candidates, a.vla_batch) < 1:
        p.error("Candidate count and batch size must be positive")
    adapter = Pi05Adapter.load(a.checkpoint, a.config)
    obs = load_observation(a.observation)
    seed = 17
    noise = np.random.default_rng(seed).standard_normal(
        (a.candidates, adapter.horizon, adapter.noise_dim), dtype=np.float32
    )
    expected = np.stack(
        [
            np.asarray(adapter.policy.infer(obs, noise=chunk[None])["actions"])
            for chunk in noise
        ]
    )
    actual = adapter.propose(obs, a.candidates, a.vla_batch, seed)
    if not np.isfinite(actual).all():
        raise ValueError("OpenPI produced non-finite actions")
    np.testing.assert_allclose(actual, expected, atol=2e-3, rtol=1e-2)
    print(
        "Native inference parity passed. Maximum absolute difference:",
        float(np.max(abs(actual - expected))),
    )
    print(
        "Candidate tensor:",
        actual.shape,
        "| distinct chunks:",
        len(np.unique(actual, axis=0)),
    )
