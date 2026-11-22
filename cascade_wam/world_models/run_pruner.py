"""Generate π0.5 candidates, optionally rank them with a trained light WM."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .light_world_model import Pruner, save_result
from .pi05_adapter import Pi05Adapter, load_observation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--observation", required=True)
    p.add_argument("--checkpoint", default="gs://openpi-assets/checkpoints/pi05_libero")
    p.add_argument("--config", default="pi05_libero")
    p.add_argument(
        "--pruner",
        help="Trained .pt; omit to save unranked candidates for data collection",
    )
    p.add_argument("--output", default="survivors.npz")
    p.add_argument("--candidates", type=int, default=256)
    p.add_argument("--keep", type=int, default=16)
    p.add_argument("--vla-batch", type=int, default=4)
    p.add_argument("--score-batch", type=int, default=64)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--uncertainty-bonus", type=float, default=0.0)
    p.add_argument("--dedup-rms", type=float, default=0.0)
    p.add_argument("--vla-device", default="cuda")
    p.add_argument(
        "--wm-device",
        default="cpu",
        help="CPU leaves GPU memory to π0.5; CUDA is optional",
    )
    a = p.parse_args()
    if a.candidates < 1:
        p.error("--candidates must be positive")
    if a.pruner and not 1 <= a.keep <= a.candidates:
        p.error("Require 1 <= keep <= candidates when ranking")
    pruner = Pruner.load(a.pruner, device=a.wm_device) if a.pruner else None
    adapter = Pi05Adapter.load(a.checkpoint, a.config, a.vla_device, a.steps)
    obs = load_observation(a.observation)
    if pruner and pruner.feature_id != adapter.feature_id:
        raise ValueError(
            "Use the same VLA checkpoint/backend/config as the pruner training features"
        )
    start = time.perf_counter()
    candidates = adapter.propose(obs, a.candidates, a.vla_batch, a.seed)
    generation_s = time.perf_counter() - start
    features = adapter.encode(obs)
    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if pruner is None:
        np.savez_compressed(
            output,
            actions=candidates,
            **features,
            feature_id=np.array(adapter.feature_id),
            prompt=np.array(obs["prompt"]),
        )
        print(
            f"Saved {len(candidates)} UNRANKED candidates to {output}; no quality claim."
        )
        return
    start = time.perf_counter()
    result = pruner.rank(
        features,
        candidates,
        feature_id=adapter.feature_id,
        keep=a.keep,
        batch_size=a.score_batch,
        uncertainty_bonus=a.uncertainty_bonus,
        dedup_rms=a.dedup_rms,
    )
    save_result(
        result,
        output,
        {
            "feature_id": adapter.feature_id,
            "seed": a.seed,
            "generation_s": generation_s,
            "pruning_s": time.perf_counter() - start,
            "uncertainty_bonus": a.uncertainty_bonus,
            "dedup_rms": a.dedup_rms,
            "prompt": obs["prompt"],
            "horizon": adapter.horizon,
            "outcome_protocol": pruner.metadata["outcome_protocol"],
        },
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "actions_shape": list(result["actions"].shape),
                "indices": result["indices"].tolist(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
