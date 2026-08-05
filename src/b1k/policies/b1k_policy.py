"""B1K Policy Transforms

Transforms BEHAVIOR-1K observations to model format.

Reference: https://github.com/wensi-ai/openpi/blob/behavior/src/openpi/policies/b1k_policy.py
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

# Deliberately NOT imported from omnigibson: the pinned BEHAVIOR-1K submodule ships the
# 2025 (256-D) table, which would silently mis-slice the 2026 (61-D) data whenever the
# simulator happens to be installed. The vendored module carries both, keyed by width.
from b1k.shared.proprio_constants import (
    PROPRIO_SPECS_BY_WIDTH,
    STATE_ACTION_DIM,
)


def make_b1k_example() -> dict:
    """Creates a random input example for the Droid policy."""
    return {
        "observation/egocentric_camera": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_right": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.random.rand(23),
        "prompt": "do something",
    }

# Physical gripper travel, used to map raw width onto the action space's [-1, 1].
# From the 2025 q99 statistics; re-check against per-year norm stats if grippers look
# saturated at one end.
MAX_GRIPPER_WIDTH = 0.1


def extract_state_from_proprio(proprio_data):
    """Reduce raw R1Pro proprioception to the 23-D vector the model consumes.

    Layout matches the official ACTION_QPOS_INDICES so state and action align
    dimension-for-dimension:
        [0:3]   base_qvel
        [3:7]   trunk_qpos      (torso)
        [7:14]  arm_left_qpos
        [14]    left gripper    (normalized to [-1, 1])
        [15:22] arm_right_qpos
        [22]    right gripper   (normalized to [-1, 1])

    The index table is chosen from the trailing dimension (61 -> 2026, 256 -> 2025)
    because the two years' layouts overlap in field names but not offsets, so guessing
    wrong yields plausible-looking garbage instead of an error. Input that is already
    reduced to 23 dims is passed through unchanged; anything else raises.

    We assume perfect correlation for the two gripper fingers (their qpos is summed).
    """
    width = np.shape(proprio_data)[-1]

    if width == STATE_ACTION_DIM:
        # Already the reduced vector (e.g. a pre-baked dataset). Nothing to do.
        return np.asarray(proprio_data)

    spec = PROPRIO_SPECS_BY_WIDTH.get(width)
    if spec is None:
        raise ValueError(
            f"Unsupported proprioception width {width}: expected {STATE_ACTION_DIM} "
            f"(already reduced) or one of {sorted(PROPRIO_SPECS_BY_WIDTH)} "
            f"(2026 / 2025 raw R1Pro proprio)."
        )
    idx = spec["R1Pro"]

    base_qvel = proprio_data[..., idx["base_qvel"]]            # 3
    trunk_qpos = proprio_data[..., idx["trunk_qpos"]]          # 4
    arm_left_qpos = proprio_data[..., idx["arm_left_qpos"]]    # 7
    arm_right_qpos = proprio_data[..., idx["arm_right_qpos"]]  # 7

    # Raw gripper widths -> [-1, 1] to match the action space.
    left_gripper_raw = proprio_data[..., idx["gripper_left_qpos"]].sum(axis=-1, keepdims=True)
    right_gripper_raw = proprio_data[..., idx["gripper_right_qpos"]].sum(axis=-1, keepdims=True)
    left_gripper_width = 2.0 * (left_gripper_raw / MAX_GRIPPER_WIDTH) - 1.0
    right_gripper_width = 2.0 * (right_gripper_raw / MAX_GRIPPER_WIDTH) - 1.0

    return np.concatenate([
        base_qvel,
        trunk_qpos,
        arm_left_qpos,
        left_gripper_width,
        arm_right_qpos,
        right_gripper_width,
    ], axis=-1)


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class B1kInputs(transforms.DataTransformFn):
    # Determines which model will be used (not actually used in B1K, kept for compatibility)
    model_type: _model.ModelType | str = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:

        # Handles both raw widths (61 = 2026, 256 = 2025) and an already-reduced 23-D
        # vector; raises on anything unexpected rather than mis-slicing silently.
        state = extract_state_from_proprio(data["observation/state"])
        if "actions" in data:
            action =  data["actions"]

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        base_image = _parse_image(data["observation/egocentric_camera"])
        wrist_image_left = _parse_image(data["observation/wrist_image_left"])
        wrist_image_right = _parse_image(data["observation/wrist_image_right"])

        # For B1K, always use 3 cameras (base, left_wrist, right_wrist)
        names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        images = (base_image, wrist_image_left, wrist_image_right)
        image_masks = (np.True_, np.True_, np.True_)

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            inputs["actions"] = action

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
            
        # Preserve task_index for PI_BEHAVIOR model
        if "task_index" in data:
            inputs["task_index"] = data["task_index"]
            
        # Preserve tokenized_prompt for PI_BEHAVIOR model
        if "tokenized_prompt" in data:
            inputs["tokenized_prompt"] = data["tokenized_prompt"]
        if "tokenized_prompt_mask" in data:
            inputs["tokenized_prompt_mask"] = data["tokenized_prompt_mask"]
            
        # Preserve subtask_state for PI_BEHAVIOR model
        if "subtask_state" in data:
            inputs["subtask_state"] = data["subtask_state"]
            
        # Preserve timestamp and episode_index for subtask state computation
        if "timestamp" in data:
            inputs["timestamp"] = data["timestamp"]
        if "episode_index" in data:
            inputs["episode_index"] = data["episode_index"]
            
        # Preserve initial_actions for inpainting
        if "initial_actions" in data:
            initial_actions = data["initial_actions"]
            # Pad initial_actions from 23 dimensions to 32 dimensions (model's action_dim)
            if initial_actions.shape[-1] < 32:
                padding_dim = 32 - initial_actions.shape[-1]
                padding = np.zeros(initial_actions.shape[:-1] + (padding_dim,))
                initial_actions = np.concatenate([initial_actions, padding], axis=-1)
            inputs["initial_actions"] = initial_actions

        return inputs


@dataclasses.dataclass(frozen=True)
class B1kOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Return actions (truncated to 23 dims) and preserve subtask predictions
        result = {"actions": np.asarray(data["actions"][:, :23])}
        
        # Preserve subtask prediction fields for PI_BEHAVIOR models
        if "subtask_logits" in data:
            result["subtask_logits"] = data["subtask_logits"]
        if "predicted_stage" in data:
            result["predicted_stage"] = data["predicted_stage"]
            
        return result
