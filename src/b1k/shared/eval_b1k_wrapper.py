"""B1K policy wrapper with action compression, rolling inpainting, and stage voting."""

import logging
import numpy as np
import torch
import dataclasses
from collections import deque

from openpi_client.base_policy import BasePolicy
from openpi_client.image_tools import resize_with_pad
from b1k.policies.b1k_policy import extract_state_from_proprio
from b1k.shared.correction_rules import apply_correction_rules, check_gripper_variation

logger = logging.getLogger(__name__)

RESIZE_SIZE = 224


@dataclasses.dataclass
class B1KWrapperConfig:
    """Configuration for B1K policy wrapper execution parameters."""
    actions_to_execute: int = 26
    actions_to_keep: int = 4
    execute_in_n_steps: int = 20
    history_len: int = 3
    votes_to_promote: int = 2
    time_threshold_inpaint: float = 0.3
    num_steps: int = 20
    apply_eval_tricks: bool = True


class B1KPolicyWrapper():
    """B1K policy wrapper for PI_BEHAVIOR models with action compression, rolling inpainting, and stage voting."""
    
    def __init__(
        self, 
        policy: BasePolicy,
        text_prompt: str = "PI_BEHAVIOR model (task-conditioned)",  # Not used, kept for compatibility
        action_horizon: int = 30,
        task_id: int | None = None,
        config: B1KWrapperConfig = None,
        checkpoint_switcher = None,
        max_lang_tokens: int = 0,
        task_prompts: dict | None = None,
        progress_horizon: int = 1500,
    ) -> None:
        self.base_policy = policy
        self.policy = policy
        self.checkpoint_switcher = checkpoint_switcher
        self.text_prompt = text_prompt
        self.action_horizon = action_horizon
        self.config = config if config is not None else B1KWrapperConfig()
        
        # Validate configuration
        if self.config.actions_to_execute + self.config.actions_to_keep > self.action_horizon:
            raise ValueError(
                f"actions_to_execute + actions_to_keep exceeds action_horizon"
            )
        
        # PI_BEHAVIOR specific (always True for B1K)
        self.task_id = task_id
        # Raw skill id fed as tokenized_prompt[1]. 0 = unknown, the same sentinel training
        # used for frames outside any annotated segment.
        self.current_skill_id = 0
        # Prompt width must match the checkpoint: 0 = no-language arm.
        self.max_lang_tokens = max_lang_tokens
        # task_index -> rich description, only needed by the language arm.
        self.task_prompts = task_prompts or {}
        # Denominator for subtask_progress. Training used the true episode length; at eval
        # that is unknown, so the step budget stands in.
        self.progress_horizon = progress_horizon
        self.prediction_history = deque([], maxlen=self.config.history_len)
        
        # Control loop variables
        self.last_actions = None
        self.action_index = 0
        self.step_count = 0
        self.prediction_count = 0
        self.next_initial_actions = None
    
    def reset(self):
        """Reset policy state."""
        self.policy.reset()
        self.last_actions = None
        self.action_index = 0
        self.step_count = 0
        self.prediction_count = 0
        self.next_initial_actions = None
        self.current_skill_id = 0
        self.prediction_history.clear()
        logger.info(f"Policy reset - Task ID: {self.task_id}, Action horizon: {self.action_horizon}")
    
    def _handle_task_change(self, new_task_id):
        """Handle task ID change by switching checkpoint and resetting state."""
        if self.task_id != new_task_id:
            old_task_id = self.task_id
            self.task_id = new_task_id
            
            logger.info(f"Task change: {old_task_id} → {new_task_id}")
            
            if self.checkpoint_switcher:
                new_policy = self.checkpoint_switcher.get_policy_for_task(new_task_id)
                if new_policy is not self.policy:
                    logger.info(f"📦 Switching checkpoint: task {old_task_id} → {new_task_id}")
                    self.base_policy = new_policy
                    self.policy = new_policy
                    self.policy.reset()
            
            self.current_skill_id = 0
            self.prediction_history.clear()
            self.last_actions = None
            self.action_index = 0
            self.next_initial_actions = None

    def process_obs(self, obs: dict) -> dict:
        """Process observation to match model input format."""
        prop_state = obs["robot_r1::proprio"]
        
        head_original = obs["robot_r1::robot_r1:zed_link:Camera:0::rgb"][..., :3]
        left_original = obs["robot_r1::robot_r1:left_realsense_link:Camera:0::rgb"][..., :3]
        right_original = obs["robot_r1::robot_r1:right_realsense_link:Camera:0::rgb"][..., :3]
        
        # Resize images
        head_resized = resize_with_pad(head_original, RESIZE_SIZE, RESIZE_SIZE)
        left_resized = resize_with_pad(left_original, RESIZE_SIZE, RESIZE_SIZE)
        right_resized = resize_with_pad(right_original, RESIZE_SIZE, RESIZE_SIZE)
        
        return {
            "observation/egocentric_camera": head_resized,
            "observation/wrist_image_left": left_resized,
            "observation/wrist_image_right": right_resized,
            "observation/state": prop_state,
            "prompt": self.text_prompt,
        }
    
    def update_current_stage(self, predicted_subtask_logits):
        """Track the current skill from the model's own predictions.

        Replaces the 2025 stage machine. That version promoted a monotonic counter through
        per-task time slices (TASK_NUM_STAGES), which no longer corresponds to anything the
        model learned: B1a made stages a *global* 34-entry skill vocabulary read from the
        annotations, and skills recur and revisit rather than advancing in order.

        So: majority vote over the raw predictions, no ordering assumption. Requires
        `votes_to_promote` agreeing samples in the recent history before switching, which
        keeps single-frame prediction noise from flipping the conditioning every step.

        predicted_subtask_logits is over the *dense* 34-way index; SKILL_IDX_TO_ID converts
        the winner back to the raw id that tokenized_prompt[1] carries.
        """
        from b1k.models.pi_behavior_config import SKILL_IDX_TO_ID, SKILL_ID_TO_NAME

        if self.task_id is None:
            return

        dense_idx = int(np.argmax(predicted_subtask_logits))
        raw_id = SKILL_IDX_TO_ID.get(dense_idx, 0)
        self.prediction_history.append(raw_id)

        if len(self.prediction_history) < self.config.history_len:
            return

        counts = {}
        for v in self.prediction_history:
            counts[v] = counts.get(v, 0) + 1
        winner, votes = max(counts.items(), key=lambda kv: kv[1])

        if votes >= self.config.votes_to_promote and winner != self.current_skill_id:
            old = self.current_skill_id
            self.current_skill_id = winner
            self.prediction_history.clear()
            logger.info(
                f"skill {old} ({SKILL_ID_TO_NAME.get(old, '?')}) → "
                f"{winner} ({SKILL_ID_TO_NAME.get(winner, '?')})  "
                f"task {self.task_id} step {self.step_count} votes {votes}/{self.config.history_len}"
            )

    def prepare_batch_for_pi_behavior(self, batch):
        """Build the model inputs exactly as training produced them.

        The previous version emitted [task_id, current_stage] plus a "subtask_state" field.
        Neither matches what the model now consumes, and both failures are quiet or fatal:

        - tokenized_prompt[1] must be the *raw* skill id (sparse in 1..102); SKILL_ID_LUT
          maps raw -> dense inside the model. current_stage was a monotonic 0,1,2 counter,
          so it indexed skill_embeddings at the wrong row without any error.
        - subtask_progress was never supplied at all, and the model raises
          "subtask_progress must be provided in Observation for PI_BEHAVIOR (B1a stages)".
        - the language arm needs 2 + max_lang_tokens entries, rebuilt with the same wording
          training used (b1k.transforms.compose_prompt), from the *predicted* skill.

        Width follows the checkpoint's own config, so one wrapper serves both A/B arms:
        max_lang_tokens == 0 -> [task_id, skill_id]; otherwise + text tokens.
        """
        from b1k.models.pi_behavior_config import SKILL_ID_TO_NAME, UNKNOWN_SKILL_NAME

        task_id = self.task_id if self.task_id is not None else -1
        batch_copy = batch.copy()
        batch_copy.pop("prompt", None)

        skill_id = int(self.current_skill_id)
        prompt = [task_id, skill_id]
        mask = [True, True]

        if self.max_lang_tokens > 0:
            from b1k.transforms import compose_prompt, _get_tokenizer

            desc = self.task_prompts.get(task_id, f"task {task_id}")
            name = SKILL_ID_TO_NAME.get(skill_id, UNKNOWN_SKILL_NAME)
            ids, m = _get_tokenizer(self.max_lang_tokens).tokenize(compose_prompt(desc, name))
            prompt += list(np.asarray(ids, dtype=np.int32))
            mask += list(np.asarray(m, dtype=bool))

        batch_copy["tokenized_prompt"] = np.array(prompt, dtype=np.int32)
        batch_copy["tokenized_prompt_mask"] = np.array(mask, dtype=bool)
        # Training used current_frame / task_duration. At eval the true duration is unknown,
        # so this is an estimate from the step budget -- it is a conditioning signal, not a
        # label, and the model saw the full [0, 1] range during training.
        batch_copy["subtask_progress"] = np.float32(
            min(1.0, self.step_count / max(1, self.progress_horizon))
        )
        return batch_copy
    
    def _interpolate_actions(self, actions, target_steps):
        """Interpolate actions using cubic spline."""
        from scipy.interpolate import interp1d
        
        original_indices = np.linspace(0, len(actions)-1, len(actions))
        target_indices = np.linspace(0, len(actions)-1, target_steps)
        
        interpolated = np.zeros((target_steps, actions.shape[1]))
        for dim in range(actions.shape[1]):
            f = interp1d(original_indices, actions[:, dim], kind='cubic')
            interpolated[:, dim] = f(target_indices)
        
        return interpolated

    def act(self, obs: dict) -> torch.Tensor:
        """Main action function."""
        
        # Extract task_id from observations
        if "task_id" in obs:
            new_task_id = int(obs["task_id"][0])
            self._handle_task_change(new_task_id)
        
        raw_state = obs["robot_r1::proprio"]
        current_state = extract_state_from_proprio(raw_state)
        
        # Check if we need new actions
        if self.last_actions is None or self.action_index >= self.config.execute_in_n_steps:
            
            # Process observation
            model_input = self.process_obs(obs)
            model_input = self.prepare_batch_for_pi_behavior(model_input)
            
            # Add rolling inpainting if available
            if self.next_initial_actions is not None and ("initial_actions" not in model_input or model_input["initial_actions"] is None):
                model_input["initial_actions"] = self.next_initial_actions
            
            # Get prediction
            if "initial_actions" in model_input and model_input["initial_actions"] is not None:
                output = self.policy.infer(model_input, initial_actions=model_input["initial_actions"])
            else:
                output = self.policy.infer(model_input)
            
            actions = output["actions"]
            
            # Ensure correct shape
            if len(actions.shape) == 3:
                actions = actions[0]
            if actions.shape[1] > 23:
                actions = actions[:, :23]
            
            # Apply eval tricks if enabled
            should_compress = self.config.execute_in_n_steps < self.config.actions_to_execute
            
            if self.config.apply_eval_tricks:
                if self.task_id is not None:
                    actions_before = actions.copy()
                    actions, corrected_stage = apply_correction_rules(
                        self.task_id, self.current_skill_id, current_state, actions
                    )
                    
                    # Log if stage was corrected
                    if corrected_stage != self.current_skill_id:
                        logger.info(f"correction rule changed skill {self.current_skill_id} → {corrected_stage} (task {self.task_id}, step {self.step_count})")
                        self.current_skill_id = corrected_stage
                        self.prediction_history.clear()
                    
                    # Log if actions were modified
                    if not np.allclose(actions_before, actions, rtol=1e-3):
                        max_diff = np.max(np.abs(actions_before - actions))
                        logger.info(f"correction rule modified actions (max diff {max_diff:.4f}, task {self.task_id}, skill {self.current_skill_id})")
                
                if should_compress:
                    has_high_variation, mean_var, max_var = check_gripper_variation(
                        actions, self.config.actions_to_execute
                    )
                    if has_high_variation:
                        should_compress = False
                        logger.info(f"🔧 Gripper variation: Compression disabled (mean: {mean_var:.4f}, max: {max_var:.4f})")
            
            # Determine execution parameters
            actions_to_execute = self.config.actions_to_execute if should_compress else self.config.execute_in_n_steps
            execute_steps = self.config.execute_in_n_steps
            
            # Save actions for next inpainting (before compression)
            inpainting_start = actions_to_execute
            inpainting_end = inpainting_start + self.config.actions_to_keep
            
            if len(actions) >= inpainting_end:
                self.next_initial_actions = actions[inpainting_start:inpainting_end].copy()
            else:
                self.next_initial_actions = None
            
            # Extract and compress actions
            self.last_actions = actions[:actions_to_execute].copy()
            
            if should_compress:
                compressed_actions = self._interpolate_actions(self.last_actions, execute_steps)
                compression_factor = actions_to_execute / execute_steps
                compressed_actions[:, :3] *= compression_factor  # Scale velocities
                self.last_actions = compressed_actions
            
            self.action_index = 0
            self.prediction_count += 1
            
            # Log prediction details (at lower frequency, every 10 predictions)
            if self.prediction_count % 10 == 0:
                compression_status = f"compressed {actions_to_execute}→{execute_steps}" if should_compress else f"uncompressed ({execute_steps})"
                logger.info(f"🎯 Prediction #{self.prediction_count} | Actions: {compression_status} | Inpainting: {self.next_initial_actions is not None}")
            
            # Update stage based on model predictions
            if "subtask_logits" in output:
                self.update_current_stage(output["subtask_logits"])
        
        # Get current action from sequence
        if self.action_index >= len(self.last_actions):
            self.action_index = 0
            
        current_action = self.last_actions[self.action_index]
        self.action_index += 1
        self.step_count += 1
        
        # Log progress every 100 steps
        if self.step_count % 100 == 0:
            logger.info(f"step {self.step_count} | task {self.task_id} | skill {self.current_skill_id} | predictions {self.prediction_count}")
        
        # Convert to torch tensor
        action_tensor = torch.from_numpy(current_action).float()
        if len(action_tensor) > 23:
            action_tensor = action_tensor[:23]
        
        return action_tensor

