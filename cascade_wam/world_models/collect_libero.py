"""Collect matched-start π0.5 candidates and real LIBERO completion labels.

Reconstructs each branch by reset + initial state + exact action-prefix replay.
This avoids assuming MuJoCo qpos/qvel alone restores controller/episode state.
Does not execute hardware actions.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .light_world_model import canonical_actions
from .pi05_adapter import Pi05Adapter


def observation_for_policy(obs, prompt):
    from openpi_client import image_tools

    quat = np.array(obs["robot0_eef_quat"], dtype=np.float64, copy=True)
    w = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1 - w * w)
    aa = np.zeros(3) if np.isclose(den, 0.0) else quat[:3] * (2 * np.arccos(w) / den)

    def image(key):
        # Same 180-degree orientation and resize as OpenPI's LIBERO example.
        img = np.ascontiguousarray(obs[key][::-1, ::-1])
        return image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))

    return {
        "observation/image": image("agentview_image"),
        "observation/wrist_image": image("robot0_eye_in_hand_image"),
        "observation/state": np.concatenate(
            [obs["robot0_eef_pos"], aa, obs["robot0_gripper_qpos"]]
        ).astype(np.float32),
        "prompt": str(prompt),
    }


def reset_replay(env, initial, prefix, seed):
    env.seed(seed)
    env.reset()
    obs = env.set_init_state(np.asarray(initial))
    for _ in range(10):
        obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
    for action in prefix:
        obs, _, _, _ = env.step(action.tolist())
    return obs


def load_initial_states(path):
    """Read LIBERO's NumPy arrays with the restricted PyTorch unpickler."""
    import torch

    try:
        from numpy._core.multiarray import _reconstruct
    except ImportError:  # NumPy 1.x uses numpy.core.
        from numpy.core.multiarray import _reconstruct

    allowed = [
        _reconstruct,
        (_reconstruct, "numpy.core.multiarray._reconstruct"),
        np.ndarray,
        np.dtype,
        type(np.dtype("float32")),
        type(np.dtype("float64")),
    ]
    with torch.serialization.safe_globals(allowed):
        states = torch.load(path, map_location="cpu", weights_only=True)
    states = np.asarray(states)
    if (
        states.ndim != 2
        or not np.issubdtype(states.dtype, np.floating)
        or not np.isfinite(states).all()
    ):
        raise ValueError(f"Invalid LIBERO initial states in {path}")
    return states


def verify_replay(env, sim_state, observation, expected_observation):
    """Check the simulator and the actual policy inputs before branching."""
    if not np.allclose(env.get_sim_state(), sim_state, atol=1e-7, rtol=1e-7):
        raise RuntimeError("Branch replay changed the simulator state")
    if not np.allclose(
        observation["observation/state"],
        expected_observation["observation/state"],
        atol=1e-7,
        rtol=1e-7,
    ):
        raise RuntimeError("Branch replay changed the policy proprioception")
    for key in ("observation/image", "observation/wrist_image"):
        if not np.array_equal(observation[key], expected_observation[key]):
            raise RuntimeError(f"Branch replay changed {key}")


def execute_until_success(env, obs, chunk):
    success = bool(env.check_success())
    used = 0
    for action in chunk:
        if success:
            break
        obs, _, done, _ = env.step(action.tolist())
        used += 1
        success = bool(env.check_success())
        if done and not success:
            raise RuntimeError(
                "Simulator horizon ended before branch budget; increase environment horizon"
            )
    return obs, success, used


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="gs://openpi-assets/checkpoints/pi05_libero")
    p.add_argument("--config", default="pi05_libero")
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--tasks", type=int, nargs="+", default=[0])
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--start-episode", type=int, default=0)
    p.add_argument("--prefix-steps", type=int, nargs="+", default=[0, 20, 40])
    p.add_argument("--candidates", type=int, default=8)
    p.add_argument("--vla-batch", type=int, default=2)
    p.add_argument("--continuation-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="branch_data")
    a = p.parse_args()
    if (
        min(a.episodes, a.candidates, a.vla_batch) < 1
        or min(a.prefix_steps) < 0
        or min(a.tasks) < 0
        or a.start_episode < 0
        or a.seed < 0
        or a.continuation_steps < 0
    ):
        p.error("Invalid collection sizes")
    root = Path(a.output)
    root.mkdir(parents=True, exist_ok=True)
    if any(root.glob("*.npz")):
        p.error("Use a fresh output directory to avoid overwriting collected pools")

    gl_backend = os.environ.setdefault("MUJOCO_GL", "egl")
    if gl_backend in {"egl", "osmesa"}:
        os.environ.setdefault("PYOPENGL_PLATFORM", gl_backend)
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark(a.suite)()
    if len(set(a.tasks)) != len(a.tasks) or max(a.tasks) >= suite.n_tasks:
        p.error("Select distinct task indices from the suite")
    adapter = Pi05Adapter.load(a.checkpoint, a.config)
    for task_id in a.tasks:
        task = suite.get_task(task_id)
        initial_states = load_initial_states(
            Path(get_libero_path("init_states"))
            / task.problem_folder
            / task.init_states_file
        )
        if a.start_episode + a.episodes > len(initial_states):
            raise ValueError("Requested episodes exceed available initial states")
        env = OffScreenRenderEnv(
            bddl_file_name=str(
                Path(get_libero_path("bddl_files"))
                / task.problem_folder
                / task.bddl_file
            ),
            camera_heights=256,
            camera_widths=256,
            horizon=max(
                1000, max(a.prefix_steps) + adapter.horizon + a.continuation_steps + 20
            ),
        )
        low, high = env.env.action_spec
        controller = {
            "low": np.asarray(low).tolist(),
            "high": np.asarray(high).tolist(),
            "clip": True,
        }
        meta = {
            "feature_id": adapter.feature_id,
            "feature_spec": adapter.feature_spec,
            "demo": False,
            "controller": controller,
            "outcome_protocol": {
                "kind": "full_candidate_chunk_then_frozen_pi05",
                "candidate_horizon": adapter.horizon,
                "continuation_steps": a.continuation_steps,
                "continuation_replan_steps": adapter.horizon,
                "terminate_on_success": True,
                "denoising_steps": 10,
                "continuation_noise": "paired_seed_by_episode_and_prefix",
            },
        }
        try:
            for ep in range(a.start_episode, a.start_episode + a.episodes):
                initial = initial_states[ep]
                ep_seed = a.seed + task_id * 10000 + ep
                episode_id = f"{a.suite}/task{task_id}/init{ep}/seed{ep_seed}"
                # Build one reference action prefix per source episode.
                obs = reset_replay(env, initial, [], ep_seed)
                prefix = []
                for target in sorted(set(a.prefix_steps)):
                    terminated = False
                    while len(prefix) < target:
                        po = observation_for_policy(obs, task.language)
                        chunk = adapter.propose(po, 1, 1, ep_seed + len(prefix))[0]
                        chunk, valid, _ = canonical_actions(chunk[None], **controller)
                        if not valid[0]:
                            raise RuntimeError("Non-finite reference action")
                        for action in chunk[0][: target - len(prefix)]:
                            obs, _, _, _ = env.step(action.tolist())
                            prefix.append(action.copy())
                            if env.check_success():
                                terminated = True
                                break
                        if terminated:
                            break
                    if terminated or env.check_success():
                        break
                    reference_state = np.array(env.get_sim_state(), copy=True)
                    po = observation_for_policy(obs, task.language)
                    features = adapter.encode(po)
                    proposed = adapter.propose(
                        po, a.candidates, a.vla_batch, ep_seed + 100000 + target
                    )
                    candidates, valid, _ = canonical_actions(proposed, **controller)
                    group_id = f"{episode_id}/step{target}"
                    if not valid.any():
                        raise RuntimeError(f"{group_id}: no finite candidate actions")
                    rows = []
                    for candidate_id in np.flatnonzero(valid):
                        branch_obs = reset_replay(env, initial, prefix, ep_seed)
                        verify_replay(
                            env,
                            reference_state,
                            observation_for_policy(branch_obs, task.language),
                            po,
                        )
                        branch_obs, success, candidate_steps = execute_until_success(
                            env, branch_obs, candidates[candidate_id]
                        )
                        future = adapter.encode(
                            observation_for_policy(branch_obs, task.language)
                        )["z"]
                        elapsed = 0
                        while not success and elapsed < a.continuation_steps:
                            follow = adapter.propose(
                                observation_for_policy(branch_obs, task.language),
                                1,
                                1,
                                ep_seed + 200000 + target * 1000 + elapsed,
                            )
                            follow, follow_valid, _ = canonical_actions(
                                follow, **controller
                            )
                            if not follow_valid[0]:
                                raise RuntimeError("Non-finite continuation action")
                            follow = follow[0][: a.continuation_steps - elapsed]
                            branch_obs, success, used = execute_until_success(
                                env, branch_obs, follow
                            )
                            elapsed += used
                        rows.append(
                            {
                                **features,
                                "future": future,
                                "actions": candidates[candidate_id],
                                "success": float(success),
                                "episode": episode_id,
                                "branch_group": group_id,
                                "candidate_id": int(candidate_id),
                                "candidate_steps": candidate_steps,
                                "continuation_steps_used": elapsed,
                            }
                        )
                        print(
                            json.dumps(
                                {
                                    "group": group_id,
                                    "candidate": int(candidate_id),
                                    "success": success,
                                }
                            ),
                            flush=True,
                        )
                    if rows:
                        name = f"{a.suite}_t{task_id}_ep{ep}_seed{a.seed}_step{target}"
                        path = root / f"{name}.npz"
                        np.savez_compressed(
                            path,
                            **{k: np.stack([r[k] for r in rows]) for k in rows[0]},
                            metadata=np.array(json.dumps(meta)),
                        )
                        np.savez_compressed(
                            root / f"{name}.observation.npz",
                            image=po["observation/image"],
                            wrist_image=po["observation/wrist_image"],
                            state=po["observation/state"],
                            prompt=np.array(po["prompt"]),
                        )
                    # Restore the reference before extending its prefix.
                    obs = reset_replay(env, initial, prefix, ep_seed)
        finally:
            env.close()


if __name__ == "__main__":
    main()
