"""Rerank saved π0.5 candidates with trained light and medium latent predictors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .light_world_model import Pruner


def compatible(light, medium):
    if light.feature_id != medium.feature_id:
        raise ValueError("Light and medium feature encoders differ")
    for k in ("controller", "outcome_protocol", "demo"):
        if light.metadata.get(k) != medium.metadata.get(k):
            raise ValueError(f"Light and medium disagree about {k}")
    for k in ("latent_dim", "tokens", "goal_dim", "state_dim", "action_dim", "horizon"):
        if getattr(light.models[0].config, k) != getattr(medium.models[0].config, k):
            raise ValueError(f"Light and medium input dimension mismatch: {k}")


def cascade(
    light, medium, features, actions, feature_id, keep=16, final_keep=1, batch_size=32
):
    compatible(light, medium)
    if not 1 <= final_keep <= keep <= len(actions):
        raise ValueError("Require 1 <= final_keep <= keep <= number of candidates")
    first = light.rank(
        features, actions, feature_id=feature_id, keep=keep, batch_size=batch_size
    )
    if len(first["actions"]) < final_keep:
        raise ValueError("Too few valid candidates for final_keep")
    second = medium.rank(
        features,
        first["actions"],
        feature_id=feature_id,
        keep=final_keep,
        batch_size=batch_size,
    )
    final_indices = first["indices"][second["indices"]]
    heavy = np.full(len(actions), np.nan, dtype=np.float32)
    heavy[first["indices"]] = second["all_success_mean"]
    return {
        "actions": second["actions"],
        "indices": final_indices,
        "survivor_actions": first["actions"],
        "survivor_indices": first["indices"],
        "light_scores": first["all_success_mean"],
        "medium_scores": heavy,
        "final_scores": second["scores"],
        "all_actions": first["all_actions"],
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--candidates",
        required=True,
        help="Unranked output of run_pruner.py; contains z/state/goal/actions/feature_id",
    )
    p.add_argument("--light", required=True)
    p.add_argument("--medium", required=True)
    p.add_argument("--output", default="cascade_survivors.npz")
    p.add_argument("--keep", type=int, default=16)
    p.add_argument("--final-keep", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    light, medium = [Pruner.load(f, device=a.device) for f in (a.light, a.medium)]
    with np.load(a.candidates, allow_pickle=False) as f:
        features = {k: f[k] for k in ("z", "state", "goal")}
        actions, feature_id = f["actions"], str(f["feature_id"].item())
    out = cascade(
        light, medium, features, actions, feature_id, a.keep, a.final_keep, a.batch_size
    )
    path = Path(a.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "candidates": len(actions),
                "survivors": len(out["survivor_indices"]),
                "final_indices": out["indices"].tolist(),
                "feature_id": feature_id,
                "score": "temperature-scaled success logits; ensemble mean if multiple members",
                "medium_scores_for_pruned_candidates": "NaN means not evaluated",
                "outcome_protocol": light.metadata["outcome_protocol"],
            },
            indent=2,
        )
    )
    print(f"Saved {path}; final indices {out['indices'].tolist()}")
