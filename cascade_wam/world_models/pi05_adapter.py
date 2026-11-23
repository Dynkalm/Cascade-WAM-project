"""OpenPI π0.5 candidate sampling and frozen feature extraction.

Verified against the user-provided openpi-main source, November 18, 2026.
The small private-API boundary is deliberately isolated in this file.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np


def map_tree(fn, tree):
    if isinstance(tree, dict):
        return {k: map_tree(fn, v) for k, v in tree.items()}
    return fn(tree)


def pool_tokens(tokens, side=4):
    """Exact nonoverlapping spatial pooling of square patch grids [N,D]."""
    tokens = np.asarray(tokens, dtype=np.float32)
    n, d = tokens.shape
    grid = int(n**0.5)
    if grid * grid != n or grid % side:
        raise ValueError(f"Cannot pool {n} tokens to {side}x{side}.")
    x = tokens.reshape(side, grid // side, side, grid // side, d)
    return x.mean(axis=(1, 3)).reshape(side * side, d)


def standardize_features(x):
    x = np.asarray(x, dtype=np.float32)
    return (x - x.mean(axis=-1, keepdims=True)) / np.sqrt(
        x.var(axis=-1, keepdims=True) + 1e-6
    )


class Pi05Adapter:
    def __init__(
        self,
        policy,
        model_config,
        checkpoint_id,
        config_name="pi05_libero",
        pool_side=4,
    ):
        required = (
            "_model",
            "_input_transform",
            "_output_transform",
            "_sample_actions",
            "_is_pytorch_model",
        )
        if any(not hasattr(policy, k) for k in required):
            raise TypeError(
                "Unsupported OpenPI Policy API. Use the supplied OpenPI snapshot."
            )
        self.policy = policy
        self.config = model_config
        self.horizon = int(model_config.action_horizon)
        self.noise_dim = int(model_config.action_dim)  # 32 padded coordinates, NOT 7.
        self.pool_side = pool_side
        self.is_torch = policy._is_pytorch_model
        self.feature_spec = {
            "encoder": "openpi_pi05_projected_siglip_v1",
            "checkpoint": str(checkpoint_id),
            "config": config_name,
            "backend": "pytorch" if self.is_torch else "jax",
            "pool_side": pool_side,
            "normalization": "per_token_layernorm_eps1e-6",
            "camera_order": ["base_0_rgb", "left_wrist_0_rgb"],
            "goal": "masked_mean_input_language_embeddings_layernorm",
        }
        self.feature_id = hashlib.sha256(
            json.dumps(self.feature_spec, sort_keys=True).encode()
        ).hexdigest()
        if not self.is_torch:
            from openpi.shared import nnx_utils

            self._image_encoder = nnx_utils.module_jit(
                policy._model.PaliGemma.img.__call__, static_argnames=("train",)
            )
        else:
            policy._model.eval()

    @classmethod
    def load(
        cls,
        checkpoint="gs://openpi-assets/checkpoints/pi05_libero",
        config_name="pi05_libero",
        device="cuda",
        num_steps=10,
    ):
        from openpi.policies import policy_config
        from openpi.training import config as configs

        cfg = configs.get_config(config_name)
        if not cfg.model.pi05:
            raise ValueError("Select a π0.5 config, not π0 or π0-FAST.")
        # Avoid a lengthy torch.compile warmup on Colab.
        cfg = dataclasses.replace(
            cfg, model=dataclasses.replace(cfg.model, pytorch_compile_mode=None)
        )
        policy = policy_config.create_trained_policy(
            cfg,
            checkpoint,
            pytorch_device=device,
            sample_kwargs={"num_steps": num_steps},
        )
        return cls(policy, cfg.model, checkpoint, config_name)

    def _prepare(self, observation):
        return self.policy._input_transform(copy.deepcopy(observation))

    def _batch(self, inputs, count):
        if self.is_torch:
            import torch

            dev = self.policy._pytorch_device
            return map_tree(
                lambda x: (
                    torch.as_tensor(np.array(x), device=dev)
                    .unsqueeze(0)
                    .repeat(count, *([1] * np.ndim(x)))
                ),
                inputs,
            )
        import jax.numpy as jnp

        return map_tree(
            lambda x: jnp.repeat(jnp.asarray(x)[None], count, axis=0), inputs
        )

    @staticmethod
    def _numpy(x):
        if hasattr(x, "detach"):
            return x.detach().float().cpu().numpy()
        return np.asarray(x, dtype=np.float32)

    def propose(self, observation, count=256, batch_size=4, seed=0):
        """Return denormalized controller action chunks [K,H,D].

        Preserves upstream input/output transforms. Distinct full-size Gaussian
        noise is supplied per candidate; no jitter is added to final actions.
        Input transformation happens once; prefix KV is reused within each
        denoising call by OpenPI, but not shared across candidate microbatches.
        """
        from openpi.models import model as model_api

        if count < 1 or batch_size < 1:
            raise ValueError("count and batch_size must be positive")
        rng = np.random.default_rng(seed)
        inputs = self._prepare(observation)
        out = []
        for start in range(0, count, batch_size):
            n = min(batch_size, count - start)
            noise = rng.standard_normal(
                (n, self.horizon, self.noise_dim), dtype=np.float32
            )
            batch = self._batch(inputs, n)
            obs = model_api.Observation.from_dict(batch)
            kwargs = dict(self.policy._sample_kwargs)
            if self.is_torch:
                import torch

                noise_backend = torch.as_tensor(
                    noise, device=self.policy._pytorch_device
                )
                with torch.inference_mode():
                    raw = self.policy._sample_actions(
                        self.policy._pytorch_device,
                        obs,
                        **{**kwargs, "noise": noise_backend},
                    )
            else:
                import jax
                import jax.numpy as jnp

                raw = self.policy._sample_actions(
                    jax.random.key(seed), obs, **{**kwargs, "noise": jnp.asarray(noise)}
                )
            raw, states = self._numpy(raw), self._numpy(batch["state"])
            for j in range(n):
                transformed = self.policy._output_transform(
                    {"state": states[j], "actions": raw[j]}
                )
                out.append(np.asarray(transformed["actions"], dtype=np.float32))
        return np.stack(out)

    def encode(self, observation):
        """Frozen current visual tokens, goal vector, raw robot state.

        Reuses π0.5's projected image features, not WAVE-0's released feature
        space. Therefore WAVE-0 checkpoints cannot be loaded into our predictor.
        """
        from openpi.models import model as model_api

        inputs = self._prepare(observation)
        obs = model_api.Observation.from_dict(self._batch(inputs, 1))
        cameras = self.feature_spec["camera_order"]
        if self.is_torch:
            import torch
            from openpi.models_pytorch import preprocessing_pytorch

            with torch.inference_mode():
                obs = preprocessing_pytorch.preprocess_observation_pytorch(
                    obs, train=False
                )
                base = self.policy._model.paligemma_with_expert
                visual = [
                    self._numpy(base.embed_image(obs.images[k]))[0] for k in cameras
                ]
                words = self._numpy(base.embed_language_tokens(obs.tokenized_prompt))[0]
                mask = self._numpy(obs.tokenized_prompt_mask)[0].astype(bool)
        else:
            obs = model_api.preprocess_observation(None, obs, train=False)
            visual = [
                self._numpy(self._image_encoder(obs.images[k], train=False)[0])[0]
                for k in cameras
            ]
            words = self._numpy(
                self.policy._model.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            )[0]
            mask = np.asarray(obs.tokenized_prompt_mask)[0].astype(bool)
        z = standardize_features(
            np.concatenate([pool_tokens(v, self.pool_side) for v in visual])
        )
        goal = standardize_features(words[mask].mean(axis=0))
        state = np.asarray(observation["observation/state"], dtype=np.float32)
        return {"z": z, "goal": goal, "state": state}


def load_observation(path):
    """NPZ with image, wrist_image, state and scalar Unicode prompt; no pickle."""
    with np.load(path, allow_pickle=False) as f:
        out = {
            "observation/image": f["image"],
            "observation/wrist_image": f["wrist_image"],
            "observation/state": f["state"].astype(np.float32),
            "prompt": str(f["prompt"].item()),
        }
    for k in ["observation/image", "observation/wrist_image"]:
        if out[k].ndim != 3 or out[k].shape[-1] != 3 or out[k].dtype != np.uint8:
            raise ValueError(f"{k} must be uint8 RGB [height,width,3]")
    return out
