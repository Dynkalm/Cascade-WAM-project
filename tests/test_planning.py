from cascade_wam import ActionCandidate, HierarchicalPlanner, cascade_select


class LightweightModel:
    def __init__(self):
        self.calls = 0

    def predict(self, observation, candidate):
        self.calls += 1
        return {"feasibility": candidate.metadata["feasible"], "effectiveness": candidate.metadata["light"]}


class HeavyweightModel:
    def __init__(self):
        self.calls = 0

    def evaluate(self, observation, candidate):
        self.calls += 1
        return candidate.metadata["heavy"]


def test_planner_filters_then_evaluates_only_survivors():
    light = LightweightModel()
    heavy = HeavyweightModel()
    planner = HierarchicalPlanner(
        lambda observation: [
            ActionCandidate("a", {"feasible": 0.9, "light": 0.8, "heavy": 0.1}),
            ActionCandidate("b", {"feasible": 0.1, "light": 1.0, "heavy": 1.0}),
            ActionCandidate("c", {"feasible": 0.8, "light": 0.2, "heavy": 0.9}),
        ],
        light,
        heavy,
        survivor_limit=2,
    )

    result = planner.plan(object())

    assert [candidate.trajectory for candidate in result.evaluated_candidates] == ["a", "c"]
    assert result.selected.trajectory == "c"
    assert light.calls == 3
    assert heavy.calls == 2
    assert result.heavy_evaluation_fraction == 2 / 3


def test_light_model_rejects_infeasible_candidates():
    candidates = [
        ActionCandidate("bad", {"feasible": 0.0, "score": 1.0}),
        ActionCandidate("good", {"feasible": 1.0, "score": 0.4}),
    ]

    def light(observation, candidate):
        return candidate.metadata["feasible"], candidate.metadata["score"]

    def heavy(observation, candidate):
        return candidate.metadata["score"]

    result = cascade_select(candidates, light, heavy, survivor_limit=1)

    assert result.selected.trajectory == "good"
    assert result.heavy_model_calls == 1
    assert [candidate.trajectory for candidate in result.evaluated_candidates] == ["good"]


def test_no_feasible_candidates_returns_none():
    planner = HierarchicalPlanner(
        lambda observation: [ActionCandidate("a"), ActionCandidate("b")],
        lambda observation, candidate: (0.0, 1.0),
        lambda observation, candidate: 1.0,
    )

    result = planner.plan(object())

    assert result.selected is None
    assert result.evaluated_candidates == []
    assert result.heavy_model_calls == 0
