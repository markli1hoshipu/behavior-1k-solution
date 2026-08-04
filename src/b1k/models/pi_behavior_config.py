"""PI_BEHAVIOR Model Configuration

Configuration for PI_BEHAVIOR model on BEHAVIOR-1K challenge.
"""

import dataclasses
import json
import pathlib
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

from b1k.models.observation import Observation

if TYPE_CHECKING:
    from b1k.models.pi_behavior import PiBehavior


# Skill vocabulary derived from BEHAVIOR-1K 2026 annotations (34 unique skill_ids,
# sparse in [1..102]). Enumerated from meta/tasks.jsonl × per-episode
# annotations/task-XXXX/episode_YYYYYYYY.json → skill_annotation[i]["skill_id"].
SKILL_ID_TO_IDX = {
    1: 0,   2: 1,   3: 2,   4: 3,   5: 4,   6: 5,   8: 6,   9: 7,   10: 8,
    11: 9,  12: 10, 13: 11, 14: 12, 19: 13, 28: 14, 34: 15, 46: 16, 50: 17,
    61: 18, 67: 19, 69: 20, 70: 21, 88: 22, 90: 23, 91: 24, 92: 25, 93: 26,
    94: 27, 95: 28, 98: 29, 99: 30, 100: 31, 101: 32, 102: 33,
}
NUM_SKILLS = len(SKILL_ID_TO_IDX)  # 34
UNKNOWN_SKILL_ID = 0  # sentinel for missing / padding
# Dense lookup table: raw skill_id (0..MAX_SKILL_ID_KEY) -> idx into skill_embeddings.
# Out-of-vocab keys map to 0 (unknown). Built as a tuple so import-time is JAX-free.
_MAX_SKILL_ID_KEY = max(SKILL_ID_TO_IDX.keys()) + 1  # 103
SKILL_ID_LUT = tuple([SKILL_ID_TO_IDX.get(i, 0) for i in range(_MAX_SKILL_ID_KEY)])

# --- Deprecated (kept for import-compat with any lingering call sites; do not use) ---
TASK_NUM_STAGES = None  # type: ignore
MAX_NUM_STAGES = None  # type: ignore
TOTAL_TASK_STAGE_EMBEDDINGS = None  # type: ignore
TASK_STAGE_OFFSETS = None  # type: ignore


@dataclasses.dataclass(frozen=True)
class PiBehaviorConfig(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 30
    max_token_len: int = 200  # Only used for compatibility, not for actual tokenization
    
    # Number of tasks in the behavior dataset (2026 challenge = 100 tasks)
    num_tasks: int = 100
    # Task embedding dimension - will match the paligemma width
    task_embedding_dim: int = None  # type: ignore
    # Number of unique skill ids for skill-annotation-based stage embedding (B1a)
    num_skills: int = NUM_SKILLS
    # Max # of PaliGemma text tokens for the rich language prompt (Path B hybrid)
    max_lang_tokens: int = 128

    # Rich-prompt source: HF meta/tasks.jsonl-style file mapping task_index -> task string
    task_prompts_path: str = "b1k/2026-challenge-demos/meta/tasks.jsonl"
    
    # Whether to use correlated noise matching action covariance structure
    # Requires correlation matrix in norm_stats (computed by compute_norm_stats.py)
    use_correlated_noise: bool = True
    
    # Shrinkage parameter for correlation regularization
    # Applied as: S_regularized = beta * S + (1-beta) * I
    # beta=1.0 means full correlation (no shrinkage)
    # beta=0.7 means 70% correlation + 30% independence (recommended for robustness)
    # beta=0.0 means independence (no correlation)
    correlation_beta: float = 0.5
    
    # FAST auxiliary training configuration
    use_fast_auxiliary: bool = False  # Enable FAST during training
    fast_loss_weight: float = 0.1  # Weight for FAST loss (vs flow loss)
    
    # Action dimensions to encode with FAST (default: 0:6, 7:23 = 22 dims)
    # Format: "0:6,7:23" or list of tuples [(0, 6), (7, 23)]
    fast_encoded_dims: str | list[tuple[int, int]] = "0:6,7:23"
    
    # FAST tokenizer vocab size
    fast_vocab_size: int = 1024
    
    # Max FAST tokens to predict (truncate if exceeded)
    max_fast_tokens: int = 32
    
    # FAST tokenizer path (set during initialization, relative to assets_dir/asset_id)
    fast_tokenizer_path: str | None = None
    
    # KV cache transformation for cross-layer attention between VLM and action expert
    # Allows each action expert layer to attend to a learned combination of all VLM layers
    use_kv_transform: bool = True
    
    # Knowledge insulation: stop action expert gradients from flowing to VLM backbone
    # VLM trains on FAST tokens only, action expert on flow matching with frozen VLM features
    # Implements approach from https://www.physicalintelligence.company/research/knowledge_insulation
    use_knowledge_insulation: bool = True
    
    # Subtask/stage prediction auxiliary loss weight (relative to action loss)
    # Higher values emphasize stage prediction accuracy at the expense of action quality
    subtask_loss_weight: float = 0.1
    
    # Time threshold for inpainting during inference
    # Stop enforcing inpainting constraint when t < threshold (let model be free in final steps)
    time_threshold_inpaint: float = 0.3
    
    # Vision backbone finetuning control
    freeze_vision_backbone: bool = True

    def __post_init__(self):
        if self.task_embedding_dim is None:
            paligemma_config = _gemma.get_config(self.paligemma_variant)
            object.__setattr__(self, "task_embedding_dim", paligemma_config.width)
    
    def get_fast_dim_ranges(self) -> list[tuple[int, int]]:
        """Parse fast_encoded_dims into list of ranges."""
        if isinstance(self.fast_encoded_dims, str):
            ranges = []
            for range_str in self.fast_encoded_dims.split(','):
                start, end = map(int, range_str.strip().split(':'))
                ranges.append((start, end))
            return ranges
        return self.fast_encoded_dims
    
    def get_total_fast_dims(self) -> int:
        """Get total number of dimensions encoded by FAST."""
        return sum(end - start for start, end in self.get_fast_dim_ranges())

    @property
    @override
    def model_type(self):
        return "pi_behavior"

    @override
    def create(self, rng: at.KeyArrayLike) -> "PiBehavior":
        from b1k.models.pi_behavior import PiBehavior

        return PiBehavior(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple["Observation", _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            obs_kwargs = {
                "images": {
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                "image_masks": {
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                "state": jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                # tokenized_prompt = [task_id, skill_id, lang_tok_0, ..., lang_tok_{L-1}]
                #   pos 0: task_id (0..num_tasks-1) routed to task_embeddings
                #   pos 1: raw skill_id routed via SKILL_ID_LUT to skill_embeddings
                #   pos 2..: PaliGemma text tokens (padded)
                "tokenized_prompt": jax.ShapeDtypeStruct([batch_size, 2 + self.max_lang_tokens], jnp.int32),
                "tokenized_prompt_mask": jax.ShapeDtypeStruct([batch_size, 2 + self.max_lang_tokens], bool),
                # Continuous progress = current_frame / episode_length in [0, 1]
                "subtask_progress": jax.ShapeDtypeStruct([batch_size], jnp.float32),
            }
            
            if self.use_fast_auxiliary:
                obs_kwargs["fast_tokens"] = jax.ShapeDtypeStruct([batch_size, self.max_fast_tokens], jnp.int32)
                obs_kwargs["fast_token_mask"] = jax.ShapeDtypeStruct([batch_size, self.max_fast_tokens], bool)
            
            observation_spec = Observation(**obs_kwargs)
        
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)
        return observation_spec, action_spec