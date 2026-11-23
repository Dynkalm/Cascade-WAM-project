"""Trainable latent dynamics, outcome scoring, and candidate pruning.

WAVE-0-inspired architecture; independently implemented, no released weights.
Dynamics do not see the goal: physical consequences should depend on state and
actions. The success head DOES see the goal and the predicted future.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


@dataclass
class WorldConfig:
    latent_dim: int = 2048
    goal_dim: int = 2048
    tokens: int = 32
    state_dim: int = 8
    action_dim: int = 7
    horizon: int = 10
    width: int = 256
    depth: int = 4
    heads: int = 8


class ActionBlock(nn.Module):
    """Action-conditioned adaptive-LayerNorm residual transformer block."""

    def __init__(self, width, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(width, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(width, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True, dropout=0.0)
        self.mlp = nn.Sequential(
            nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width)
        )
        self.modulate = nn.Sequential(nn.SiLU(), nn.Linear(width, 6 * width))
        nn.init.zeros_(self.modulate[-1].weight)
        nn.init.zeros_(self.modulate[-1].bias)

    def forward(self, x, action):
        shift, scale, gate, shift2, scale2, gate2 = self.modulate(action).chunk(6, -1)
        y = self.norm1(x) * (1 + scale[:, None]) + shift[:, None]
        x = x + gate[:, None] * self.attn(y, y, y, need_weights=False)[0]
        y = self.norm2(x) * (1 + scale2[:, None]) + shift2[:, None]
        return x + gate2[:, None] * self.mlp(y)


class LightWorldModel(nn.Module):
    def __init__(self, config: WorldConfig):
        super().__init__()
        c = self.config = config
        if min(asdict(c).values()) < 1 or c.width % c.heads:
            raise ValueError("Invalid world model dimensions")
        self.visual = nn.Linear(c.latent_dim, c.width)
        self.state = nn.Linear(c.state_dim, c.width)
        self.action = nn.Linear(c.action_dim, c.width)
        self.action_pos = nn.Parameter(torch.randn(1, c.horizon, c.width) * 0.02)
        self.spatial_pos = nn.Parameter(torch.randn(1, c.tokens, c.width) * 0.02)
        self.future_queries = nn.Parameter(torch.randn(1, c.tokens, c.width) * 0.02)
        layer = nn.TransformerEncoderLayer(
            c.width,
            c.heads,
            4 * c.width,
            dropout=0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.action_encoder = nn.TransformerEncoder(
            layer, 2, norm=nn.LayerNorm(c.width), enable_nested_tensor=False
        )
        self.blocks = nn.ModuleList(
            [ActionBlock(c.width, c.heads) for _ in range(c.depth)]
        )
        self.delta_out = nn.Sequential(
            nn.LayerNorm(c.width), nn.Linear(c.width, c.latent_dim)
        )
        self.value_visual = nn.Linear(c.latent_dim, c.width)
        self.value_goal = nn.Linear(c.goal_dim, c.width)
        self.value_attention = nn.MultiheadAttention(
            c.width, c.heads, batch_first=True, dropout=0
        )
        self.value_head = nn.Sequential(
            nn.LayerNorm(2 * c.width),
            nn.Linear(2 * c.width, c.width),
            nn.GELU(),
            nn.Linear(c.width, 1),
        )

    def predict(self, z, state, actions):
        c = self.config
        if tuple(actions.shape[1:]) != (c.horizon, c.action_dim):
            raise ValueError(
                f"Expected actions [B,{c.horizon},{c.action_dim}], got {actions.shape}"
            )
        if tuple(z.shape[1:]) != (c.tokens, c.latent_dim):
            raise ValueError("Feature layout differs from training")
        a = self.action_encoder(self.action(actions) + self.action_pos)
        x = torch.cat(
            [
                self.visual(z) + self.spatial_pos,
                self.state(state)[:, None],
                a,
                self.future_queries.expand(len(z), -1, -1) + self.spatial_pos,
            ],
            1,
        )
        for block in self.blocks:
            x = block(x, a.mean(1))
        return z + self.delta_out(x[:, -c.tokens :])

    def value(self, future, goal):
        # No direct action shortcut: action ranking must use predicted consequences.
        v = self.value_visual(future) + self.spatial_pos
        g = self.value_goal(goal)[:, None]
        pooled = self.value_attention(g, v, v, need_weights=False)[0][:, 0]
        return self.value_head(torch.cat([pooled, g[:, 0]], -1)).squeeze(-1)

    def forward(self, z, state, actions, goal):
        future = self.predict(z, state, actions)
        return future, self.value(future, goal)


def canonical_actions(actions, low, high, clip=True):
    """Enforce the selected controller's numeric semantics before scoring.

    Clipping is appropriate only when the real controller does the same. It is
    not a collision/reachability/physical feasibility test. Never score unclipped
    actions and then execute their clipped versions.
    """
    a = np.asarray(actions, dtype=np.float32)
    lo, hi = np.asarray(low, dtype=np.float32), np.asarray(high, dtype=np.float32)
    if a.ndim != 3 or lo.shape != (a.shape[-1],) or hi.shape != lo.shape:
        raise ValueError("actions must be [K,H,D]; bounds must have D entries")
    if not np.isfinite(lo).all() or not np.isfinite(hi).all() or np.any(hi <= lo):
        raise ValueError("Controller bounds must be finite with high > low")
    finite = np.isfinite(a).all(axis=(1, 2))
    outside = ((a < lo) | (a > hi)).mean(axis=(1, 2))
    valid = finite & (True if clip else outside == 0)
    return np.clip(a, lo, hi) if clip else a.copy(), valid, outside


def select_survivors(
    actions, mean, std, valid, keep, uncertainty_bonus=0.0, dedup_rms=0.0
):
    """Stable top-M; optional optimism and near-duplicate removal.

    Positive uncertainty_bonus favors uncertain candidates for further checking;
    this is a heuristic, not a calibrated confidence bound. Default: mean only.
    Action distance is measured in the training-normalized action coordinates.
    """
    n = len(actions)
    if not 1 <= keep <= n or uncertainty_bonus < 0 or dedup_rms < 0:
        raise ValueError("Invalid pruning settings")
    mean, std = np.asarray(mean), np.asarray(std)
    eligible = np.asarray(valid, dtype=bool) & np.isfinite(mean) & np.isfinite(std)
    scores = mean + uncertainty_bonus * std
    order = np.argsort(-np.where(eligible, scores, -np.inf), kind="stable")
    selected = []
    for i in order:
        if not eligible[i]:
            continue
        if dedup_rms > 0 and any(
            np.sqrt(np.mean((actions[i] - actions[j]) ** 2)) <= dedup_rms
            for j in selected
        ):
            continue
        selected.append(int(i))
        if len(selected) == keep:
            break
    return np.asarray(selected, dtype=np.int64), np.where(eligible, scores, -np.inf)


class Pruner:
    def __init__(
        self,
        models,
        normalization,
        feature_id,
        metadata,
        device="cpu",
        *,
        dataset_fingerprint=None,
    ):
        self.models = [m.to(device).eval() for m in models]
        self.normalization = normalization
        self.feature_id = feature_id
        self.metadata = metadata
        self.device = device
        self.dataset_fingerprint = dataset_fingerprint
        self.temperatures = [1.0] * len(models)

    @classmethod
    def load(cls, path, device="cpu", allow_demo=False):
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        if ckpt.get("trained_steps", 0) <= 0:
            raise ValueError("Refusing to rank with untrained weights")
        if ckpt.get("demo", False) and not allow_demo:
            raise ValueError("Synthetic demo weights cannot rank real robot actions")
        models = []
        for state in ckpt["models"]:
            m = LightWorldModel(WorldConfig(**ckpt["config"]))
            m.load_state_dict(state, strict=True)
            models.append(m)
        if not models:
            raise ValueError("Checkpoint has no models")
        result = cls(
            models,
            ckpt["normalization"],
            ckpt["feature_id"],
            ckpt["metadata"],
            device,
            dataset_fingerprint=ckpt.get("dataset_fingerprint"),
        )
        result.temperatures = ckpt.get("temperatures", [1.0] * len(models))
        if len(result.temperatures) != len(models) or any(
            t <= 0 or not np.isfinite(t) for t in result.temperatures
        ):
            raise ValueError("Invalid calibration temperatures")
        return result

    @torch.inference_mode()
    def rank(
        self,
        features,
        actions,
        *,
        feature_id,
        keep=16,
        batch_size=64,
        uncertainty_bonus=0.0,
        dedup_rms=0.0,
    ):
        if feature_id != self.feature_id:
            raise ValueError(
                "Encoder/checkpoint/preprocessing mismatch; re-encode with the training setup"
            )
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        spec = self.metadata["controller"]
        canonical, valid, clip_fraction = canonical_actions(actions, **spec)
        stats = self.normalization
        a = (canonical - np.asarray(stats["action_mean"])) / np.asarray(
            stats["action_std"]
        )
        s = (features["state"] - np.asarray(stats["state_mean"])) / np.asarray(
            stats["state_std"]
        )
        if not all(np.isfinite(x).all() for x in (features["z"], features["goal"], s)):
            raise ValueError("Non-finite observation features")
        estimates = np.full((len(self.models), len(a)), np.nan, dtype=np.float32)

        def tensor(x):
            return torch.as_tensor(x, dtype=torch.float32, device=self.device)

        good = np.flatnonzero(valid)
        for start in range(0, len(good), batch_size):
            ids = good[start : start + batch_size]
            z = tensor(features["z"])[None].expand(len(ids), -1, -1)
            g = tensor(features["goal"])[None].expand(len(ids), -1)
            st = tensor(s)[None].expand(len(ids), -1)
            for j, model in enumerate(self.models):
                _, logits = model(z, st, tensor(a[ids]), g)
                estimates[j, ids] = (
                    (logits / self.temperatures[j]).sigmoid().cpu().numpy()
                )
        mean, std = estimates.mean(0), estimates.std(0)
        selected, score = select_survivors(
            a, mean, std, valid, keep, uncertainty_bonus, dedup_rms
        )
        if not len(selected):
            raise ValueError(
                "No finite, controller-valid candidates survived; no fallback action was invented"
            )
        return {
            "actions": canonical[selected],
            "indices": selected,
            "scores": score[selected],
            "success_mean": mean[selected],
            "success_std": std[selected],
            "all_actions": canonical,
            "all_scores": score,
            "all_success_mean": mean,
            "all_success_std": std,
            "valid": valid,
            "clip_fraction": clip_fraction,
        }


def save_result(result, output, metadata=None):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **result)
    selected = result["indices"].tolist()
    report = {
        "candidate_count": len(result["all_actions"]),
        "survivor_count": len(selected),
        "selected_indices": selected,
        "scores": result["scores"].tolist(),
        "score_definition": "mean predicted completion + uncertainty_bonus * ensemble std",
        "probabilities_calibrated": False,
        **(metadata or {}),
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2))
