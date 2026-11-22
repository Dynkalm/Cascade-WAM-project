"""Hierarchical filtering and evaluation of proposed robotic actions."""

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence


@dataclass(frozen=True)
class ActionCandidate:
    """A proposed action trajectory."""

    trajectory: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateScore:
    """A world-model estimate for one candidate."""

    feasibility: float
    effectiveness: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def feasible(self) -> bool:
        return self.feasibility >= 0.5


@dataclass(frozen=True)
class PlanResult:
    """The result of one hierarchical planning step."""

    selected: Optional[ActionCandidate]
    evaluated_candidates: Sequence[ActionCandidate]
    lightweight_scores: Mapping[Any, CandidateScore]
    heavyweight_scores: Mapping[Any, CandidateScore]
    light_model_calls: int
    heavy_model_calls: int

    @property
    def proposed_count(self) -> int:
        return len(self.lightweight_scores)

    @property
    def heavy_evaluation_fraction(self) -> float:
        if not self.proposed_count:
            return 0.0
        return self.heavy_model_calls / float(self.proposed_count)

    @property
    def heavy_evaluations(self) -> int:
        return self.heavy_model_calls

    @property
    def lightweight_evaluations(self) -> int:
        return self.light_model_calls


def _as_candidate(value: Any) -> ActionCandidate:
    if isinstance(value, ActionCandidate):
        return value
    if isinstance(value, Mapping):
        return ActionCandidate(
            value.get("trajectory", value.get("actions", value)),
            {k: v for k, v in value.items() if k not in ("trajectory", "actions")},
        )
    return ActionCandidate(value)


def _call_model(model: Any, candidate: ActionCandidate, observation: Any) -> Any:
    if callable(model):
        result = model(observation, candidate)
        if isinstance(result, Exception):
            raise result
        return result

    for method_name in ("predict", "evaluate", "score"):
        method = getattr(model, method_name, None)
        if callable(method):
            try:
                return method(observation, candidate)
            except TypeError:
                return method(candidate)
    raise TypeError("World model must be callable or provide predict/evaluate/score")


def _normalize_score(value: Any, *, default_feasibility: float = 0.5) -> CandidateScore:
    if isinstance(value, CandidateScore):
        return value
    if isinstance(value, Mapping):
        feasibility = value.get(
            "feasibility", value.get("feasible", default_feasibility)
        )
        if isinstance(feasibility, bool):
            feasibility = 1.0 if feasibility else 0.0
        effectiveness = value.get(
            "effectiveness", value.get("score", value.get("value", 0.0))
        )
        metadata = value.get("metadata", {})
        return CandidateScore(float(feasibility), float(effectiveness), metadata)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) >= 2:
            return CandidateScore(float(value[0]), float(value[1]))
        if len(value) == 1:
            return CandidateScore(float(value[0]), float(value[0]))
    if isinstance(value, (int, float)):
        number = float(value)
        return CandidateScore(1.0, number)
    return CandidateScore(default_feasibility, 0.0)


def _candidate_key(candidate: ActionCandidate) -> Any:
    if isinstance(candidate.trajectory, (str, int, float, bytes, tuple)):
        return candidate.trajectory
    return id(candidate)


class HierarchicalPlanner:
    """Two-stage candidate filtering followed by heavyweight ranking."""

    def __init__(
        self,
        proposal_generator: Any,
        lightweight_world_model: Any,
        heavyweight_world_model: Any,
        *,
        survivor_limit: Optional[int] = None,
        feasibility_threshold: float = 0.5,
        top_k: Optional[int] = None,
    ) -> None:
        if survivor_limit is not None and survivor_limit <= 0:
            raise ValueError("survivor_limit must be positive")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive")
        self.proposal_generator = proposal_generator
        self.lightweight_world_model = lightweight_world_model
        self.heavyweight_world_model = heavyweight_world_model
        self.survivor_limit = survivor_limit
        self.feasibility_threshold = feasibility_threshold
        self.top_k = top_k

    def generate(self, observation: Any, candidate_count: Optional[int] = None) -> Sequence[ActionCandidate]:
        if callable(self.proposal_generator):
            try:
                proposals = (
                    self.proposal_generator(observation, candidate_count)
                    if candidate_count is not None
                    else self.proposal_generator(observation)
                )
            except TypeError:
                proposals = (
                    self.proposal_generator(observation, num_samples=candidate_count)
                    if candidate_count is not None
                    else self.proposal_generator(observation)
                )
        else:
            for method_name in ("generate", "propose", "sample"):
                method = getattr(self.proposal_generator, method_name, None)
                if callable(method):
                    try:
                        proposals = (
                            method(observation, candidate_count)
                            if candidate_count is not None
                            else method(observation)
                        )
                    except TypeError:
                        proposals = (
                            method(observation, num_samples=candidate_count)
                            if candidate_count is not None
                            else method(observation)
                        )
                    break
            else:
                raise TypeError(
                    "proposal_generator must be callable or provide generate/propose/sample"
                )
        return [_as_candidate(value) for value in proposals]

    def plan(self, observation: Any, candidate_count: Optional[int] = None) -> PlanResult:
        candidates = self.generate(observation, candidate_count)
        lightweight_scores = {}
        light_model_calls = 0

        for candidate in candidates:
            raw_score = _call_model(
                self.lightweight_world_model, candidate, observation
            )
            score = _normalize_score(raw_score)
            lightweight_scores[_candidate_key(candidate)] = score
            light_model_calls += 1

        feasible = [
            candidate
            for candidate in candidates
            if lightweight_scores[_candidate_key(candidate)].feasibility
            >= self.feasibility_threshold
        ]
        feasible.sort(
            key=lambda candidate: (
                lightweight_scores[_candidate_key(candidate)].effectiveness,
                lightweight_scores[_candidate_key(candidate)].feasibility,
            ),
            reverse=True,
        )

        limit = self.survivor_limit or len(feasible)
        survivors = feasible[:limit]
        heavyweight_scores = {}

        for candidate in survivors:
            raw_score = _call_model(
                self.heavyweight_world_model, candidate, observation
            )
            score = _normalize_score(raw_score)
            heavyweight_scores[_candidate_key(candidate)] = score

        ranked = list(survivors)
        ranked.sort(
            key=lambda candidate: (
                heavyweight_scores[_candidate_key(candidate)].effectiveness,
                heavyweight_scores[_candidate_key(candidate)].feasibility,
            ),
            reverse=True,
        )
        result_limit = self.top_k or 1
        selected = ranked[0] if ranked and result_limit > 0 else None

        return PlanResult(
            selected=selected,
            evaluated_candidates=survivors,
            lightweight_scores=lightweight_scores,
            heavyweight_scores=heavyweight_scores,
            light_model_calls=light_model_calls,
            heavy_model_calls=len(survivors),
        )

    def select(self, observation: Any, candidate_count: Optional[int] = None) -> PlanResult:
        return self.plan(observation, candidate_count)


def cascade_select(
    candidates: Iterable[Any],
    lightweight_world_model: Any,
    heavyweight_world_model: Any,
    observation: Any = None,
    *,
    survivor_limit: Optional[int] = None,
    feasibility_threshold: float = 0.5,
    top_k: Optional[int] = None,
) -> PlanResult:
    """Convenience function for selecting from an existing candidate iterable."""

    class CandidateList:
        @staticmethod
        def generate(_observation: Any) -> Sequence[ActionCandidate]:
            return [_as_candidate(value) for value in candidates]

    planner = HierarchicalPlanner(
        CandidateList.generate,
        lightweight_world_model,
        heavyweight_world_model,
        survivor_limit=survivor_limit,
        feasibility_threshold=feasibility_threshold,
        top_k=top_k,
    )
    return planner.plan(observation)
