"""A hierarchical candidate-action planning architecture."""

from .planning import (
    ActionCandidate,
    CandidateScore,
    HierarchicalPlanner,
    PlanResult,
    cascade_select,
)

__all__ = [
    "ActionCandidate",
    "CascadePlanner",
    "CandidateScore",
    "HierarchicalPlanner",
    "PlanResult",
    "cascade_select",
]

# A concise name for users who prefer the architecture's family name.
CascadePlanner = HierarchicalPlanner
