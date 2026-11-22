# Cascade-WAM-project

Cascade-WAM is a small, dependency-free reference implementation of hierarchical
robotic planning:

1. A pretrained VLA policy proposes multiple action trajectories.
2. A lightweight world model estimates feasibility and effectiveness and keeps a
   bounded set of the strongest survivors.
3. A heavyweight world model evaluates only those survivors and ranks the final
   action.

## Usage

```python
from cascade_wam import ActionCandidate, HierarchicalPlanner

planner = HierarchicalPlanner(
    proposal_generator=lambda observation: [
        ActionCandidate([0.1, 0.2]),
        ActionCandidate([0.2, 0.1]),
    ],
    lightweight_world_model=lambda observation, candidate: (0.9, 0.7),
    heavyweight_world_model=lambda observation, candidate: (0.95, 0.8),
    survivor_limit=1,
)

result = planner.plan({"camera": "observation"})
if result.selected is not None:
    execute(result.selected.trajectory)
```

Scores can be `CandidateScore` objects, mappings with `feasibility` and
`effectiveness`, `(feasibility, effectiveness)` pairs, or scalar scores.
The convenience function `cascade_select(candidates, lightweight, heavyweight)`
is available for ranking an existing candidate list.

## Development

```bash
python -m pytest -q
```