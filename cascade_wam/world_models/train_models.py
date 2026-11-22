"""Train a light or medium latent world model on cached π0.5 branch outcomes.

Defaults: light=256x4, medium=512x8, both endpoint prediction. No Wan weights,
video decoder, or frozen VLA needs to reside in memory during this training.
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .light_world_model import LightWorldModel, WorldConfig
from .prepare_training import FeatureData

PRESETS = {
    "light": {"width": 256, "depth": 4, "heads": 8},
    "medium": {"width": 512, "depth": 8, "heads": 8},
}


def atomic_save(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def pairwise_loss(logits, labels, groups):
    # Positive > negative only within exactly matched observations/goals.
    # Average per group, so a large pool does not overwhelm smaller ones.
    terms = []
    for g in torch.unique(groups):
        pos = logits[(groups == g) & (labels == 1)]
        neg = logits[(groups == g) & (labels == 0)]
        if pos.numel() and neg.numel():
            terms.append(F.softplus(-(pos[:, None] - neg[None, :])).mean())
    return torch.stack(terms).mean() if terms else logits.sum() * 0


def objective(model, b, rank_weight=0.2, warmup=False):
    pred = model.predict(b["z"], b["state"], b["actions"])
    # Compute regressions in fp32 even when attention uses autocast.
    cosine = (1 - F.cosine_similarity(pred.float(), b["future"].float(), dim=-1)).mean()
    delta = F.smooth_l1_loss(pred.float() - b["z"], b["future"] - b["z"])
    # Success gradients cannot rewrite the predicted physics to encode labels.
    fake_logits = model.value(pred.detach(), b["goal"]).float()
    real_logits = model.value(b["future"], b["goal"]).float()
    bce = 0.5 * (
        F.binary_cross_entropy_with_logits(fake_logits, b["success"])
        + F.binary_cross_entropy_with_logits(real_logits, b["success"])
    )
    ranking = pairwise_loss(fake_logits, b["success"], b["group"])
    value_weight = 0.0 if warmup else 1.0
    loss = cosine + 0.5 * delta + value_weight * (bce + rank_weight * ranking)
    return loss, {
        "future_cosine_loss": float(cosine.detach()),
        "delta_huber": float(delta.detach()),
        "success_bce": float(bce.detach()),
        "pairwise": float(ranking.detach()),
    }


def tensor_batch(data, ids, device):
    return {
        k: torch.as_tensor(
            v, device=device, dtype=torch.long if k == "group" else torch.float32
        )
        for k, v in data.batch(ids).items()
    }


def probability(logits, temperature=1.0):
    x = np.clip(np.asarray(logits, dtype=np.float64) / temperature, -60, 60)
    return 1 / (1 + np.exp(-x))


def binary_metrics(labels, p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return {
        "bce": float(-np.mean(labels * np.log(p) + (1 - labels) * np.log1p(-p))),
        "brier": float(np.mean((p - labels) ** 2)),
        "positive_rate": float(labels.mean()),
    }


@torch.inference_mode()
def predict(model, data, ids, device="cpu", batch_size=32):
    model.eval()
    logits = []
    cos, total_delta, persistence, count = 0.0, 0.0, 0.0, 0
    for start in range(0, len(ids), batch_size):
        b = tensor_batch(data, ids[start : start + batch_size], device)
        pred, score = model(b["z"], b["state"], b["actions"], b["goal"])
        logits.extend(score.float().cpu().tolist())
        n = len(score)
        cos += n * float((1 - F.cosine_similarity(pred, b["future"], dim=-1)).mean())
        total_delta += n * float(F.smooth_l1_loss(pred - b["z"], b["future"] - b["z"]))
        persistence += n * float(
            (1 - F.cosine_similarity(b["z"], b["future"], dim=-1)).mean()
        )
        count += n
    return np.asarray(logits), {
        "future_cosine_loss": cos / count,
        "delta_huber": total_delta / count,
        "copy_current_cosine_loss": persistence / count,
    }


@torch.inference_mode()
def score_actual_futures(model, data, ids, device="cpu", batch_size=32):
    """Score recorded post-action features with the same head used for prediction."""
    model.eval()
    logits = []
    for start in range(0, len(ids), batch_size):
        batch = tensor_batch(data, ids[start : start + batch_size], device)
        score = model.value(batch["future"], batch["goal"])
        logits.extend(score.float().cpu().tolist())
    return np.asarray(logits)


def fit_temperature(logits, labels):
    # One scalar, fit ONLY on the separate calibration split after model selection.
    candidates = np.exp(np.linspace(np.log(0.1), np.log(10.0), 121))
    losses = [
        np.mean(np.logaddexp(0, logits / t) - labels * logits / t) for t in candidates
    ]
    return float(candidates[int(np.argmin(losses))])


def train(
    cache,
    output,
    *,
    size="light",
    epochs=30,
    batch_size=32,
    members=1,
    lr=1e-4,
    weight_decay=0.01,
    rank_weight=0.2,
    warmup_epochs=2,
    patience=8,
    seed=42,
    device="cpu",
    precision="auto",
    resume=False,
    width=None,
    depth=None,
    heads=None,
    threads=2,
):
    if (
        min(epochs, batch_size, members, threads, patience) < 1
        or lr <= 0
        or rank_weight < 0
        or not 0 <= warmup_epochs < epochs
    ):
        raise ValueError(
            "Invalid training settings; epochs must exceed dynamics warmup."
        )
    torch.set_num_threads(threads)
    data = FeatureData(cache)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    run = output.parent / (output.stem + "_run")
    if (output.exists() or run.exists()) and not resume:
        raise FileExistsError(
            "Output exists. Use --resume with the same settings, or a new output path."
        )
    run.mkdir(exist_ok=True)
    dims = data.manifest["shapes"]
    preset = PRESETS[size].copy()
    preset.update(
        {
            k: v
            for k, v in [("width", width), ("depth", depth), ("heads", heads)]
            if v is not None
        }
    )
    config = WorldConfig(
        latent_dim=dims["z"][1],
        tokens=dims["z"][0],
        goal_dim=dims["goal"][0],
        state_dim=dims["state"][0],
        action_dim=dims["actions"][1],
        horizon=dims["actions"][0],
        **preset,
    )
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm not available. Select a GPU or use --device cpu.")
    if precision == "auto":
        precision = (
            ("bf16" if torch.cuda.is_bf16_supported() else "fp16")
            if dev.type == "cuda"
            else "fp32"
        )
    if dev.type != "cuda" and precision != "fp32":
        raise ValueError(
            "This trainer uses fp32 on CPU; AMP is enabled for CUDA/ROCm only."
        )
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
        precision
    ]

    def autocast():
        return (
            torch.autocast(device_type="cuda", dtype=dtype)
            if precision != "fp32"
            else nullcontext()
        )

    signature = {
        "fingerprint": data.manifest["fingerprint"],
        "config": asdict(config),
        "size": size,
        "batch_size": batch_size,
        "members": members,
        "lr": lr,
        "weight_decay": weight_decay,
        "rank_weight": rank_weight,
        "warmup_epochs": warmup_epochs,
        "patience": patience,
        "seed": seed,
        "precision": precision,
    }
    manifest_path = run / "run_config.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != signature:
        raise ValueError("Resume settings or dataset differ. Use a new output path.")
    manifest_path.write_text(json.dumps(signature, indent=2))
    tr, va, ca = (data.splits[k] for k in ("train", "validation", "calibration"))
    exports, temperatures, member_reports = [], [], []
    total_steps = 0
    for member in range(members):
        torch.manual_seed(seed + member)
        if dev.type == "cuda":
            torch.cuda.manual_seed_all(seed + member)
        model = LightWorldModel(config).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
        last, best = run / f"member_{member}_last.pt", run / f"member_{member}_best.pt"
        start, steps, best_bce, bad, history = 0, 0, float("inf"), 0, []
        if resume and last.exists():
            ck = torch.load(last, map_location="cpu", weights_only=True)
            if ck["signature"] != signature:
                raise ValueError("Checkpoint signature mismatch")
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["optimizer"])
            scaler.load_state_dict(ck["scaler"])
            start, steps, best_bce, bad, history = (
                ck["epoch"],
                ck["steps"],
                ck["best_bce"],
                ck["bad_epochs"],
                ck["history"],
            )
            torch.set_rng_state(ck["torch_rng"])
            if dev.type == "cuda" and ck["cuda_rng"] is not None:
                torch.cuda.set_rng_state_all(ck["cuda_rng"])
        for epoch in range(start, epochs):
            if bad >= patience:
                break
            t0 = time.perf_counter()
            model.train()
            sums, rows = {}, 0
            for ids in data.batches(tr, batch_size, seed + member * 100000 + epoch):
                b = tensor_batch(data, ids, dev)
                opt.zero_grad(set_to_none=True)
                with autocast():
                    loss, parts = objective(
                        model, b, rank_weight, epoch < warmup_epochs
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0, error_if_nonfinite=precision != "fp16"
                )
                scale = scaler.get_scale()
                scaler.step(opt)
                scaler.update()
                if scaler.get_scale() >= scale:
                    steps += 1
                rows += len(ids)
                for k, v in {"loss": float(loss.detach()), **parts}.items():
                    sums[k] = sums.get(k, 0.0) + v * len(ids)
            logits, dynamics = predict(model, data, va, dev, batch_size)
            metrics = binary_metrics(data.arrays["success"][va], probability(logits))
            log = {
                "member": member,
                "epoch": epoch + 1,
                "seconds": time.perf_counter() - t0,
                "train": {k: v / rows for k, v in sums.items()},
                "validation": {**metrics, **dynamics},
            }
            history.append(log)
            print(json.dumps(log), flush=True)
            if epoch >= warmup_epochs:
                if metrics["bce"] < best_bce:
                    best_bce, bad = metrics["bce"], 0
                    atomic_save(
                        {
                            "model": model.state_dict(),
                            "epoch": epoch + 1,
                            "validation": metrics,
                        },
                        best,
                    )
                else:
                    bad += 1
            atomic_save(
                {
                    "signature": signature,
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": epoch + 1,
                    "steps": steps,
                    "best_bce": best_bce,
                    "bad_epochs": bad,
                    "history": history,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all()
                    if dev.type == "cuda"
                    else None,
                },
                last,
            )
        selected = torch.load(best, map_location="cpu", weights_only=True)
        model.load_state_dict(selected["model"])
        cal_logits, _ = predict(model, data, ca, dev, batch_size)
        temp = fit_temperature(cal_logits, data.arrays["success"][ca])
        exports.append(
            {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        )
        temperatures.append(temp)
        total_steps += steps
        member_reports.append(
            {
                "member": member,
                "best_epoch": selected["epoch"],
                "temperature": temp,
                "validation_at_selection": selected["validation"],
                "history": history,
            }
        )
        del opt, model, scaler, selected
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    atomic_save(
        {
            "format_version": 2,
            "config": asdict(config),
            "models": exports,
            "normalization": data.manifest["normalization"],
            "feature_id": data.manifest["metadata"]["feature_id"],
            "metadata": data.manifest["metadata"],
            "trained_steps": total_steps,
            "demo": data.manifest["metadata"]["demo"],
            "temperatures": temperatures,
            "dataset_fingerprint": data.manifest["fingerprint"],
            "training_source_units": data.manifest["split_units"]["train"],
            "size": size,
        },
        output,
    )
    count = sum(v.numel() for v in exports[0].values())
    report = {
        "size": size,
        "parameters_per_member": count,
        "members": members,
        "config": asdict(config),
        "precision": precision,
        "training_steps": total_steps,
        "dataset_fingerprint": data.manifest["fingerprint"],
        "calibration": "temperature fitted on separate calibration split; not a guarantee of reliability",
        "member_reports": member_reports,
        "demo": data.manifest["metadata"]["demo"],
    }
    output.with_suffix(".training.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {
                "saved": str(output),
                "parameters_per_member": count,
                "temperatures": temperatures,
            }
        ),
        flush=True,
    )
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--size", choices=list(PRESETS), default="light")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--members", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--rank-weight", type=float, default=0.2)
    p.add_argument("--warmup-epochs", type=int, default=2)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--precision", choices=["auto", "fp32", "bf16", "fp16"], default="auto"
    )
    p.add_argument("--resume", action="store_true")
    for k in ("width", "depth", "heads"):
        p.add_argument("--" + k, type=int)
    train(**vars(p.parse_args()))
