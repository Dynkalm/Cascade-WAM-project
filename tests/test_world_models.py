"""Behavioral tests for data isolation, objectives, resume and cascade mappings."""

# ruff: noqa: E402
import itertools
import sys

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
from cascade_wam.world_models import run_pruner
from cascade_wam.world_models.collect_libero import load_initial_states, verify_replay
from cascade_wam.world_models.evaluate_models import (
    group_table,
    paired_effect,
    random_m_heavy,
)
from cascade_wam.world_models.light_world_model import (
    LightWorldModel,
    Pruner,
    WorldConfig,
)
from cascade_wam.world_models.prepare_training import (
    FeatureData,
    prepare,
    source_unit,
    split_units,
)
from cascade_wam.world_models.run_cascade import cascade
from cascade_wam.world_models.smoke_training import synthetic_data
from cascade_wam.world_models.train_models import (
    binary_metrics,
    fit_temperature,
    objective,
    pairwise_loss,
    probability,
    score_actual_futures,
    train,
)


def test_split_by_initial_condition_not_seed():
    ep = np.array(
        [
            f"libero/task0/init{i}/seed{s}"
            for i in range(20)
            for s in (1, 2)
            for _ in (0, 1)
        ]
    )
    y = np.tile([0.0, 1.0], 40)
    splits, units, _ = split_units(ep, y, 42)
    for a, b in itertools.combinations(units, 2):
        assert not set(units[a]) & set(units[b])
    for ids in splits.values():
        assert len(np.unique(y[ids])) == 2
    assert source_unit("libero/task0/init1/seed2") == "libero/task0/init1"


def test_ranking_only_compares_siblings():
    logits = torch.tensor([2.0, -2.0, -4.0, 4.0], requires_grad=True)
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
    groups = torch.tensor([0, 0, 1, 1])
    expected = (
        torch.nn.functional.softplus(torch.tensor(-4.0))
        + torch.nn.functional.softplus(torch.tensor(8.0))
    ) / 2
    assert torch.allclose(pairwise_loss(logits, labels, groups), expected)
    loss = pairwise_loss(logits, labels, torch.arange(4))
    loss.backward()
    assert loss.item() == 0 and logits.grad.abs().sum() == 0


def test_success_gradient_does_not_change_dynamics():
    torch.set_num_threads(2)
    m = LightWorldModel(
        WorldConfig(
            latent_dim=8,
            goal_dim=8,
            tokens=4,
            state_dim=2,
            action_dim=2,
            horizon=4,
            width=16,
            depth=1,
            heads=4,
        )
    )
    b = {
        k: torch.randn(*shape)
        for k, shape in {
            "z": (2, 4, 8),
            "state": (2, 2),
            "actions": (2, 4, 2),
            "goal": (2, 8),
            "future": (2, 4, 8),
        }.items()
    }
    b["success"] = torch.tensor([0.0, 1.0])
    b["group"] = torch.tensor([1, 1])
    loss, _ = objective(m, b, rank_weight=0)
    loss.backward()
    g0 = m.delta_out[-1].weight.grad.clone()
    m.zero_grad()
    loss, _ = objective(m, b, rank_weight=10)
    loss.backward()
    assert torch.allclose(g0, m.delta_out[-1].weight.grad)
    m.zero_grad()
    # AdaLN starts gated at zero, so action sensitivity emerges through training.
    # Verify dynamic gradient pathway rather than assert random initialization quality.
    loss, _ = objective(m, b)
    loss.backward()
    assert m.blocks[0].modulate[-1].weight.grad.abs().sum() > 0


def test_exact_random_subset_baseline():
    y = np.array([0, 1, 0, 1.0])
    h = np.array([0.7, 0.4, 0.2, 0.9])
    outcomes = [
        y[list(c)[np.argmax(h[list(c)])]] for c in itertools.combinations(range(4), 2)
    ]
    assert random_m_heavy(y, h, 2) == pytest.approx(np.mean(outcomes))
    rows = group_table(np.zeros(4), np.array(["x"] * 4), h, h, 2)
    assert rows[0]["light_retains_any_success"] is None


def test_temperature_reduces_calibration_bce():
    logits = np.array([10, 10, -10, -10.0])
    y = np.array([1, 0, 0, 1.0])
    t = fit_temperature(logits, y)
    assert (
        binary_metrics(y, probability(logits, t))["bce"]
        < binary_metrics(y, probability(logits))["bce"]
    )


def test_cascade_uses_only_survivors_and_original_indices():
    class Fake:
        feature_id = "X"
        metadata = {"controller": {}, "outcome_protocol": {}, "demo": True}
        models = [type("M", (), {"config": WorldConfig()})()]

        def __init__(self, heavy=False):
            self.heavy = heavy
            self.seen = None

        def rank(self, f, a, **kwargs):
            self.seen = a.copy()
            keep = kwargs["keep"]
            score = a[:, 0, 0] if not self.heavy else -a[:, 0, 0]
            ids = np.argsort(-score)[:keep]
            return {
                "actions": a[ids],
                "indices": ids,
                "all_success_mean": score,
                "scores": score[ids],
                "all_actions": a,
            }

    light, medium = Fake(), Fake(True)
    a = np.arange(5, dtype=np.float32)[:, None, None]
    out = cascade(light, medium, {}, a, "X", 2, 1)
    assert out["indices"].tolist() == [3]
    assert medium.seen[:, 0, 0].tolist() == [4, 3]
    assert np.isnan(out["medium_scores"][:3]).all()


def test_cache_validation_and_resume_equivalence(tmp_path):
    torch.set_num_threads(2)
    source = tmp_path / "s.npz"
    synthetic_data(source)
    prepare([source], tmp_path / "cache")
    d = FeatureData(tmp_path / "cache")
    tr = d.splits["train"]
    assert np.allclose(d.batch(tr)["state"].mean(0), 0, atol=1e-6)
    for batch in d.batches(tr, 32, 1):
        for g in np.unique(d.group_codes[batch]):
            assert (d.group_codes[batch] == g).sum() == (d.group_codes[tr] == g).sum()
    args = dict(
        width=16, depth=1, heads=4, batch_size=64, warmup_epochs=0, lr=1e-3, patience=10
    )
    train(tmp_path / "cache", tmp_path / "resumed.pt", epochs=1, **args)
    train(tmp_path / "cache", tmp_path / "resumed.pt", epochs=3, resume=True, **args)
    train(tmp_path / "cache", tmp_path / "fresh.pt", epochs=3, **args)
    a = torch.load(tmp_path / "resumed.pt", weights_only=True)
    b = torch.load(tmp_path / "fresh.pt", weights_only=True)
    for k in a["models"][0]:
        assert torch.equal(a["models"][0][k], b["models"][0][k]), k
    with pytest.raises(ValueError, match="Synthetic"):
        Pruner.load(tmp_path / "resumed.pt")
    with pytest.raises(ValueError, match="differ"):
        train(
            tmp_path / "cache",
            tmp_path / "resumed.pt",
            epochs=4,
            resume=True,
            **{**args, "lr": 0.01},
        )
    p = Pruner.load(tmp_path / "resumed.pt", allow_demo=True)
    assert p.temperatures == a["temperatures"]
    # Duplicate pools and altered encoders must fail before any cache is installed.
    with pytest.raises(ValueError):
        prepare([source, source], tmp_path / "bad")


def test_actual_future_diagnostic_compares_the_same_survivors():
    labels = np.array([0.0, 1.0, 0.0, 1.0])
    groups = np.array(["first", "first", "second", "second"])
    predicted = np.array([0.9, 0.1, 0.2, 0.8])
    actual = np.array([0.3, 0.7, 0.2, 0.8])
    rows = group_table(
        labels,
        groups,
        predicted,
        predicted,
        1,
        actual_futures={"light": actual, "medium": actual},
    )
    assert rows[0]["light_actual_retains_any_success"] == 1
    assert rows[0]["light_retains_any_success"] == 0
    assert rows[0]["medium_actual_all_top1"] == 1
    assert rows[0]["medium_actual_same_survivors_top1"] == 0
    effect = paired_effect(
        rows, np.array(["init1", "init2"]), "light_actual_top1", "light_top1"
    )
    assert effect["mean"] == 0.5


def test_actual_future_scores_use_recorded_features_and_the_same_head():
    class Features:
        def batch(self, ids):
            future = np.array([5.0, 8.0], dtype=np.float32)[ids]
            return {
                "z": np.zeros((len(ids), 1, 1)),
                "future": future[:, None, None],
                "goal": np.ones((len(ids), 1)),
            }

    class Head(torch.nn.Module):
        def value(self, future, goal):
            return future[:, 0, 0] + goal[:, 0]

    scores = score_actual_futures(Head(), Features(), [1, 0], batch_size=1)
    assert scores.tolist() == [9.0, 6.0]


def test_branch_replay_rejects_observation_drift():
    class Environment:
        def get_sim_state(self):
            return np.array([1.0, 2.0])

    env = Environment()
    expected = {
        "observation/state": np.array([1.0]),
        "observation/image": np.zeros((2, 2, 3), dtype=np.uint8),
        "observation/wrist_image": np.zeros((2, 2, 3), dtype=np.uint8),
    }
    verify_replay(env, env.get_sim_state(), expected, expected)
    changed = {
        **expected,
        "observation/wrist_image": np.ones((2, 2, 3), dtype=np.uint8),
    }
    with pytest.raises(RuntimeError, match="wrist_image"):
        verify_replay(env, env.get_sim_state(), changed, expected)
    with pytest.raises(RuntimeError, match="simulator state"):
        verify_replay(env, np.zeros(2), expected, expected)


def test_load_initial_state_arrays_with_restricted_loader(tmp_path):
    path = tmp_path / "states.pruned_init"
    states = np.arange(6, dtype=np.float64).reshape(2, 3)
    torch.save(states, path)
    np.testing.assert_array_equal(load_initial_states(path), states)


def test_small_unranked_export_ignores_survivor_limit(tmp_path, monkeypatch):
    class Adapter:
        feature_id = "encoder"

        def propose(self, observation, count, batch_size, seed):
            return np.zeros((count, 10, 7), dtype=np.float32)

        def encode(self, observation):
            return {"z": np.zeros((32, 4)), "state": np.zeros(8), "goal": np.zeros(4)}

    output = tmp_path / "pool.npz"
    monkeypatch.setattr(run_pruner.Pi05Adapter, "load", lambda *args: Adapter())
    monkeypatch.setattr(run_pruner, "load_observation", lambda path: {"prompt": "task"})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_pruner",
            "--observation",
            "observation.npz",
            "--candidates",
            "8",
            "--output",
            str(output),
        ],
    )
    run_pruner.main()
    with np.load(output, allow_pickle=False) as pool:
        assert pool["actions"].shape == (8, 10, 7)
        assert pool["feature_id"].item() == "encoder"
