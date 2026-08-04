"""Weight loaders for PI_BEHAVIOR model initialization from Pi05 checkpoints.

Reference: https://github.com/Physical-Intelligence
"""

import dataclasses
import logging
import re

import flax.traverse_util
import numpy as np
import orbax.checkpoint as ocp

import openpi.shared.array_typing as at
import openpi.shared.download as download
import openpi.models.model as _model  # for restore_params(restore_type=np.ndarray)

# Re-export base loaders from OpenPI
from openpi.training.weight_loaders import (
    WeightLoader,
    NoOpWeightLoader,
    CheckpointWeightLoader,
    _merge_params,
)

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class PiBehaviorWeightLoader(WeightLoader):
    """Loads checkpoints for PI_BEHAVIOR model.
    
    Automatically detects:
    - Pi05 checkpoint: Loads weights, preserves new PI_BEHAVIOR parameters
    - PI_BEHAVIOR checkpoint: Loads all weights directly
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # Load checkpoint as PLAIN NUMPY ARRAYS. Using ocp.PyTreeCheckpointer directly
        # would return jax arrays with orbax's inherited sharding, which downstream
        # `jax.jit(init, in_shardings=replicated_sharding)` rejects with:
        #   ValueError: Sharding passed to pjit does not match the sharding on the
        #   respective arg. Got pjit sharding: PartitionSpec(),
        #   arg sharding: PartitionSpec(None, None, 'fsdp')
        # openpi.models.model.restore_params(..., restore_type=np.ndarray) strips
        # sharding on restore, letting the training loop reshard however it wants.
        params_path = download.maybe_download(self.params_path)
        loaded_params = _model.restore_params(params_path, restore_type=np.ndarray)

        # Some checkpoints wrap under 'params'/'value' — unwrap.
        if isinstance(loaded_params, dict) and "params" in loaded_params and len(loaded_params) == 1:
            loaded_params = loaded_params["params"]
        flat_params = flax.traverse_util.flatten_dict(loaded_params)
        if all(kp[-1] == "value" for kp in flat_params if len(kp) > 0):
            flat_params = {kp[:-1]: v for kp, v in flat_params.items()}
            loaded_params = flax.traverse_util.unflatten_dict(flat_params)
        
        # Detect checkpoint type
        has_task_embeddings = 'task_embeddings' in loaded_params
        
        if has_task_embeddings:
            # PI_BEHAVIOR checkpoint (either the 50-task submission checkpoint or a
            # previous 100-task run). We shape-adapt three tables that changed with the
            # 50→100 + B1a refactor:
            #   - task_embeddings:      [50, D] → [100, D]  (copy first 50 rows)
            #   - task_stage_embeddings [596, d] → dropped (skill_embeddings is new random-init)
            #   - stage_pred_from_vlm   [D, 15] → dropped (skill_pred_from_vlm is new random-init)
            logging.info("Loading PI_BEHAVIOR checkpoint with shape-adaptation for 50→100 + B1a")
            loaded_params = _adapt_pi_behavior_shapes(loaded_params, params)
            # After adaptation, obsolete keys are removed and new-init keys remain in
            # `params`. Use missing_regex to permit the new-init entries.
            missing_regex = (
                ".*skill_embeddings.*|"
                ".*skill_pred_from_vlm.*"
            )
            return _merge_params(loaded_params, params, missing_regex=missing_regex)
        else:
            # Loading Pi05 checkpoint - preserve new PI_BEHAVIOR-specific parameters
            logging.info("Loading Pi05 checkpoint (new PI_BEHAVIOR parameters will use random init)")

            # These parameters are NEW in PI_BEHAVIOR (not in Pi05), so keep them from params (random init)
            missing_regex = (
                ".*task_embeddings.*|"
                ".*skill_pred_from_vlm.*|"
                ".*skill_embeddings.*|"
                ".*gate_sincos.*|"
                ".*gate_task_stage.*|"
                ".*gate_task.*|"
                ".*fusion_layer.*|"
                ".*stage_projection.*|"
                ".*task_subtask_fusion.*|"
                ".*fast_token_embedding.*|"
                ".*fast_token_proj.*|"
                ".*kv_transform.*"
            )
            return _merge_params(loaded_params, params, missing_regex=missing_regex)


def _adapt_pi_behavior_shapes(loaded: at.Params, model_params: at.Params) -> at.Params:
    """In-place adapt of the 50-task PI_BEHAVIOR checkpoint into the 100-task + B1a model.

    - Expands task_embeddings [50, D] → [100, D] by copying rows 0..49 from the checkpoint
      and rows 50..99 from the freshly-initialized model params (random init).
    - Drops task_stage_embeddings and stage_pred_from_vlm since their shapes differ
      irreconcilably from the new skill_embeddings / skill_pred_from_vlm tables.
    """
    flat_loaded = flax.traverse_util.flatten_dict(loaded)
    flat_model = flax.traverse_util.flatten_dict(model_params)

    # 1. Expand task_embeddings if the row count grew.
    # model_v may be abstract/sharded during setup — read shape only, don't materialize.
    # Keep result as numpy so downstream jit can shard freely.
    rng = np.random.default_rng(seed=42)
    for kp, v in list(flat_loaded.items()):
        if not any("task_embeddings" == part for part in kp) or not hasattr(v, "shape"):
            continue
        model_v = flat_model.get(kp)
        if model_v is None or not hasattr(model_v, "shape"):
            continue
        if v.shape == model_v.shape:
            continue
        if v.ndim != 2 or len(model_v.shape) != 2 or v.shape[1] != model_v.shape[1] or v.shape[0] >= model_v.shape[0]:
            continue

        n_ckpt = v.shape[0]
        n_target, d = model_v.shape
        logging.info(
            f"Expanding {'/'.join(str(p) for p in kp)} {v.shape} → {model_v.shape} "
            f"(rows 0..{n_ckpt - 1} from ckpt, rows {n_ckpt}..{n_target - 1} random init)"
        )
        # Small-scale random init for the new rows — same distribution as nnx.Embed
        # default (normal * 1/sqrt(features)). Numpy so downstream jit can reshard.
        random_new_rows = (rng.standard_normal((n_target - n_ckpt, d)) / np.sqrt(d)).astype(v.dtype)
        new_v = np.concatenate([np.asarray(v), random_new_rows], axis=0)
        flat_loaded[kp] = new_v

    # 2. Drop obsolete keys so their new counterparts stay at random init.
    obsolete_substrings = ("task_stage_embeddings", "stage_pred_from_vlm")
    to_delete = [
        kp for kp in list(flat_loaded.keys())
        if any(any(sub in part for sub in obsolete_substrings) for part in kp)
    ]
    for kp in to_delete:
        logging.info(f"Dropping obsolete checkpoint key {'/'.join(kp)} (replaced by skill_* in B1a)")
        del flat_loaded[kp]

    return flax.traverse_util.unflatten_dict(flat_loaded)
