"""Data transforms for BEHAVIOR-1K dataset.

Standard transforms imported from OpenPI.
B1K-specific: TaskIndexToTaskId, ComputeSubtaskStateFromMeta, TokenizeFASTActions

Reference: https://github.com/wensi-ai/openpi/tree/behavior
"""

import dataclasses
import logging
import numpy as np

# Import all standard transforms from OpenPI
from openpi.transforms import (
    DataTransformFn,
    DataDict,
    Group,
    CompositeTransform,
    compose,
    RepackTransform,
    Normalize,
    Unnormalize,
    ResizeImages,
    SubsampleActions,
    DeltaActions,
    AbsoluteActions,
    PadStatesAndActions,
    PromptFromLeRobotTask,  # Not used for PI_BEHAVIOR but kept for compatibility
    flatten_dict,
    unflatten_dict,
    transform_dict,
    apply_tree,
    pad_to_dim,
    make_bool_mask,
)

import glob
import hashlib
import json
import os
import pathlib

from b1k.shared.normalize import NormStats
from openpi.models.tokenizer import PaligemmaTokenizer


_ANN_INDEX_CACHE: dict[str, dict[int, dict]] = {}
_TOKENIZER_CACHE: dict[int, PaligemmaTokenizer] = {}
_TASK_PROMPTS_CACHE: dict[str, dict[int, str]] = {}


def _load_annotations_index(annotations_dir: str, episodes_per_task: int = 200) -> dict[int, dict]:
    """Build LeRobot episode_index -> annotation dict, memoized per annotations_dir.

    Assumes the 2026 layout: annotations_dir/task-XXXX/episode_YYYYYYYY.json, where the
    LeRobot episode_index of the i-th (sorted) annotation file under task-T is
    T*episodes_per_task + i. Loads all files eagerly (~50 MB for 100*200 files).
    """
    if annotations_dir in _ANN_INDEX_CACHE:
        return _ANN_INDEX_CACHE[annotations_dir]

    root = pathlib.Path(annotations_dir).expanduser()
    idx: dict[int, dict] = {}
    task_dirs = sorted(root.glob("task-*"))
    if not task_dirs:
        raise FileNotFoundError(f"No task-XXXX directories found under {annotations_dir}")
    for td in task_dirs:
        try:
            task_id = int(td.name.split("-")[1])
        except (IndexError, ValueError):
            continue
        ann_files = sorted(td.glob("episode_*.json"))
        for i, af in enumerate(ann_files):
            ep_idx = task_id * episodes_per_task + i
            with open(af) as fh:
                ann = json.load(fh)
            idx[ep_idx] = {
                "task_duration": int(ann.get("meta_data", {}).get("task_duration", 0)),
                "skill_annotation": ann.get("skill_annotation", []),
            }
    logging.info(f"Loaded {len(idx)} episode annotations from {annotations_dir}")
    _ANN_INDEX_CACHE[annotations_dir] = idx
    return idx


def _load_task_prompts(path: str) -> dict[int, str]:
    """Load meta/tasks.jsonl (task_index -> rich prompt string)."""
    if path in _TASK_PROMPTS_CACHE:
        return _TASK_PROMPTS_CACHE[path]
    result: dict[int, str] = {}
    with open(pathlib.Path(path).expanduser()) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            result[int(row["task_index"])] = row["task"]
    _TASK_PROMPTS_CACHE[path] = result
    logging.info(f"Loaded {len(result)} rich task prompts from {path}")
    return result


def _get_tokenizer(max_len: int) -> PaligemmaTokenizer:
    if max_len not in _TOKENIZER_CACHE:
        _TOKENIZER_CACHE[max_len] = PaligemmaTokenizer(max_len=max_len)
    return _TOKENIZER_CACHE[max_len]


# Sentinel used when the skill is unknown or deliberately dropped from the prompt.
UNKNOWN_SKILL_NAME = "unknown"


def compose_prompt(task_description: str, skill_name: str) -> str:
    """Canonical rich prompt: full task description + the current skill in words.

    Kept module-level (rather than inlined in the transform) so the eval path can build
    the exact same string from its *predicted* skill — training and inference must agree
    on the wording or the model sees a different distribution at eval time.

    The scaffold is emitted even when the skill is unknown, so token positions stay
    comparable between the informed and dropped cases.
    """
    return f"{task_description} Current skill: {skill_name or UNKNOWN_SKILL_NAME}"


def _skill_dropout_draw(episode_index: int, current_frame: int, seed: int) -> float:
    """Deterministic pseudo-random draw in [0, 1) keyed on (episode, frame, seed).

    Deliberately not an RNG object: these transforms run inside torch DataLoader
    workers, where numpy's *global* seed is not differentiated per worker, so
    `np.random.*` would produce the same dropout pattern in every worker. Hashing the
    sample identity instead is worker-safe and makes a run exactly reproducible.

    Trade-off: a given (episode, frame) always lands on the same side of the threshold,
    so the model never sees that exact frame both with and without its skill. With
    ~211M frames in the 2026 set there is ample coverage of both regimes; vary `seed`
    to reshuffle the split.
    """
    h = hashlib.blake2b(
        f"{int(episode_index)}:{int(current_frame)}:{int(seed)}".encode(),
        digest_size=8,
    ).digest()
    return int.from_bytes(h, "big") / float(1 << 64)


@dataclasses.dataclass(frozen=True)
class AnnotateSkillFromAnnotation(DataTransformFn):
    """B1a: emit skill_id + subtask_progress from per-episode BEHAVIOR-1K skill annotation.

    Reads annotations/task-XXXX/episode_YYYYYYYY.json (one per LeRobot episode) and:
    - locates the skill segment whose frame_duration=[start, end) contains the current
      frame, and emits its `skill_id[0]` as data["skill_id"]
    - emits data["subtask_progress"] = current_frame / episode_length in [0, 1]

    Both fields are consumed by TaskIndexToPromptTokens (skill_id → tokenized_prompt[1])
    and by the model (subtask_progress → Observation.subtask_progress).
    """

    annotations_dir: str = ""
    # Fallback episodes-per-task for the assumed LeRobot indexing scheme.
    episodes_per_task: int = 200
    # Fallback skill_id when the current frame doesn't fall in any segment (padded/edge).
    fallback_skill_id: int = 0

    def __call__(self, data: DataDict) -> DataDict:
        # Inference mode / missing metadata → emit defaults so the model doesn't crash.
        if not self.annotations_dir or "episode_index" not in data or "timestamp" not in data:
            data.setdefault("skill_id", np.array(self.fallback_skill_id, dtype=np.int32))
            data.setdefault("skill_name", "")
            data.setdefault("subtask_progress", np.array(0.0, dtype=np.float32))
            return data

        try:
            ann_idx = _load_annotations_index(self.annotations_dir, self.episodes_per_task)
        except FileNotFoundError as e:
            logging.warning(f"AnnotateSkillFromAnnotation: {e}. Emitting defaults.")
            data["skill_id"] = np.array(self.fallback_skill_id, dtype=np.int32)
            data["skill_name"] = ""
            data["subtask_progress"] = np.array(0.0, dtype=np.float32)
            return data

        ep_idx = int(data["episode_index"])
        rec = ann_idx.get(ep_idx)
        if rec is None:
            data["skill_id"] = np.array(self.fallback_skill_id, dtype=np.int32)
            data["skill_name"] = ""
            data["subtask_progress"] = np.array(0.0, dtype=np.float32)
            return data

        current_frame = float(data["timestamp"]) * 30.0  # 30 FPS
        ep_len = max(1, int(rec["task_duration"]))
        progress = float(np.clip(current_frame / ep_len, 0.0, 1.0))

        # Linear scan segments; typical episode has <100 segments so this is cheap.
        # frame_duration comes in two shapes:
        #   flat:  [start, end]              (~99.8% of segments)
        #   nested: [[s1, e1], [s2, e2], …]  (~0.2%, single skill spanning discontinuous ranges)
        skill_id = self.fallback_skill_id
        skill_name = ""
        for seg in rec["skill_annotation"]:
            fd = seg.get("frame_duration", [])
            if not fd:
                continue
            # Normalize to list of [start, end] pairs
            if isinstance(fd[0], (list, tuple)):
                ranges = fd
            elif len(fd) >= 2 and isinstance(fd[0], (int, float)):
                ranges = [fd]
            else:
                continue
            matched = False
            for r in ranges:
                if len(r) < 2:
                    continue
                start, end = r[0], r[1]
                if start <= current_frame < end:
                    ids = seg.get("skill_id", [])
                    if ids:
                        skill_id = int(ids[0])
                    # Human-readable name for the same segment ("move to", "place on",
                    # ...), used to render the skill into the language prompt.
                    descs = seg.get("skill_description", [])
                    if descs:
                        skill_name = str(descs[0])
                    matched = True
                    break
            if matched:
                break

        data["skill_id"] = np.array(skill_id, dtype=np.int32)
        data["skill_name"] = skill_name
        data["subtask_progress"] = np.array(progress, dtype=np.float32)
        return data


@dataclasses.dataclass(frozen=True)
class TaskIndexToPromptTokens(DataTransformFn):
    """Assemble tokenized_prompt = [task_id, skill_id, lang_tok_0, ..., lang_tok_{L-1}].

    - task_id from data["task_id"] (inference) or data["task_index"] (training)
    - skill_id from data["skill_id"] (populated by AnnotateSkillFromAnnotation)
    - lang tokens from PaligemmaTokenizer(compose_prompt(task_description, skill_name))

    Also forwards subtask_progress as a float scalar so the Observation dataclass picks
    it up as a top-level field (via RepackTransform).

    Skill-in-prompt dropout
    -----------------------
    The skill name is rendered into the text, but on a `skill_prompt_dropout` fraction of
    samples it is replaced by "unknown". This is not regularization for its own sake — it
    protects `skill_pred_from_vlm`. That head reads the base_task token, which sits in the
    same bidirectional attention block as the text tokens (unlike skill_embeddings, which
    lives behind the first ar_mask=True fused token and therefore cannot leak). So with
    the skill spelled out in the text, predicting it is trivial and the auxiliary loss
    collapses — yet at eval that head is exactly what supplies the skill, since no
    ground-truth annotation exists there. Dropped samples keep it honest.

    Note this drops the skill from the *text* only; tokenized_prompt[1] still carries the
    ground-truth skill_id, which is both the skill_embeddings input and the auxiliary
    target. Teacher forcing therefore remains on the embedding path; if eval shows errors
    compounding, dropping that too requires separating input from target in the model.
    """

    # Path to meta/tasks.jsonl (rows: {task_index, task_name, task}).
    task_prompts_path: str = ""
    # Optional dataset task_index -> model task_id remap. If None, identity.
    task_mapping: dict[int, int] | None = None
    # Max PaliGemma text tokens (must match PiBehaviorConfig.max_lang_tokens).
    max_lang_tokens: int = 128
    # Fraction of samples whose skill name is replaced by "unknown" in the prompt.
    # 0.0 disables (always informed); 1.0 never shows the skill in text.
    skill_prompt_dropout: float = 0.5
    # Salt for the deterministic dropout draw — change it to reshuffle which samples
    # are dropped without changing anything else.
    skill_dropout_seed: int = 0

    def __call__(self, data: DataDict) -> DataDict:
        # Task id
        if "task_id" in data:
            task_id = int(data["task_id"])
        elif "task_index" in data:
            task_index = int(data["task_index"])
            task_id = self.task_mapping[task_index] if self.task_mapping else task_index
        elif "tokenized_prompt" in data:
            return data
        else:
            raise ValueError("Need task_index, task_id, or existing tokenized_prompt.")

        skill_id = int(data.get("skill_id", 0))
        subtask_progress = float(data.get("subtask_progress", 0.0))

        # Rich language prompt lookup (falls back to a minimal string).
        if self.task_prompts_path:
            prompts = _load_task_prompts(self.task_prompts_path)
        else:
            prompts = {}
        # max_lang_tokens=0 = no-language arm: emit the width-2 prompt Ilia's model used,
        # [task_id, skill_id], and skip tokenization entirely. Building a tokenizer with
        # max_len=0 would be meaningless, and the model's embed_prefix skips the text span
        # for any prompt of width 2.
        if self.max_lang_tokens == 0:
            return {
                **{k: v for k, v in data.items() if k != "skill_name"},
                "tokenized_prompt": np.array([task_id, skill_id], dtype=np.int32),
                "tokenized_prompt_mask": np.array([True, True], dtype=bool),
                "subtask_progress": np.asarray(subtask_progress, dtype=np.float32),
            }

        task_description = prompts.get(task_id, f"task {task_id}")

        # Render the current skill into the text, dropping it on a deterministic subset.
        skill_name = str(data.get("skill_name", "") or "")
        if skill_name and self.skill_prompt_dropout > 0.0:
            draw = _skill_dropout_draw(
                episode_index=int(data.get("episode_index", 0)),
                current_frame=int(float(data.get("timestamp", 0.0)) * 30.0),
                seed=self.skill_dropout_seed,
            )
            if draw < self.skill_prompt_dropout:
                skill_name = ""
        prompt = compose_prompt(task_description, skill_name)

        tok = _get_tokenizer(self.max_lang_tokens)
        lang_tokens, lang_mask = tok.tokenize(prompt, state=None)  # [max_lang], [max_lang]

        prompt_tokens = np.concatenate([
            np.array([task_id, skill_id], dtype=np.int32),
            lang_tokens.astype(np.int32),
        ], axis=0)
        prompt_mask = np.concatenate([
            np.array([True, True], dtype=bool),
            lang_mask.astype(bool),
        ], axis=0)

        # skill_name is an intermediate only: AnnotateSkillFromAnnotation emits it so the
        # prompt can be rendered here, and it must not survive into the batch. It is a
        # Python str, which collates to a numpy unicode array and makes JAX raise
        # "Dtype <U16 is not a valid JAX array type" at device_put.
        out = {k: v for k, v in data.items() if k != "skill_name"}
        return {
            **out,
            "tokenized_prompt": prompt_tokens,
            "tokenized_prompt_mask": prompt_mask,
            "subtask_progress": np.asarray(subtask_progress, dtype=np.float32),
        }


# Backwards-compat shims for older callers:
TaskIndexToTaskId = TaskIndexToPromptTokens
ComputeSubtaskStateFromMeta = AnnotateSkillFromAnnotation


@dataclasses.dataclass(frozen=True)
class TokenizeFASTActions(DataTransformFn):
    """Tokenize actions to FAST discrete tokens for PI_BEHAVIOR auxiliary training.
    
    Converts continuous actions to discrete tokens using a trained FAST tokenizer.
    This is used for auxiliary training to improve action quality.
    
    Note: Uses GLOBAL normalization even if per-timestamp normalization is enabled
    elsewhere, to preserve temporal smoothness for better DCT compression.
    
    Args:
        tokenizer_path: Path to trained FAST tokenizer directory
        encoded_dim_ranges: List of (start, end) tuples for dimensions to encode
        max_fast_tokens: Maximum number of tokens (truncate/pad)
        norm_stats: Normalization stats for denorm/renorm if per-timestamp is used
        use_per_timestamp: Whether per-timestamp normalization is being used
    
    Assumes:
    - data["actions"] exists and is normalized
    - data["state"] exists (for delta computation)
    
    Creates:
    - data["fast_tokens"]: Discrete token indices [max_fast_tokens]
    - data["fast_token_mask"]: Boolean mask for valid tokens [max_fast_tokens]
    """
    
    # Path to FAST tokenizer
    tokenizer_path: str
    # Action dimensions to encode: [(start, end), ...]
    encoded_dim_ranges: list[tuple[int, int]]
    # Max tokens to predict
    max_fast_tokens: int = 180
    # Normalization stats (for denorm/renorm if per-timestamp is used)
    norm_stats: dict[str, NormStats] | None = None
    # Whether per-timestamp normalization is being used
    use_per_timestamp: bool = False
    
    def _get_tokenizer(self):
        """Lazy load tokenizer (called in each worker process)."""
        # Check if already loaded in this process
        if not hasattr(self, '_tokenizer'):
            import importlib.util
            from pathlib import Path
            
            tokenizer_dir = Path(self.tokenizer_path)
            processor_file = tokenizer_dir / "processing_action_tokenizer.py"
            
            if not processor_file.exists():
                raise FileNotFoundError(f"Tokenizer processing file not found at {processor_file}")
            
            # Load the module dynamically
            spec = importlib.util.spec_from_file_location("processing_action_tokenizer", processor_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Get the UniversalActionProcessor class
            UniversalActionProcessor = module.UniversalActionProcessor
            
            # Load the tokenizer using from_pretrained
            tokenizer = UniversalActionProcessor.from_pretrained(str(tokenizer_dir))
            
            # Store in this object (won't be pickled due to lazy loading pattern)
            object.__setattr__(self, '_tokenizer', tokenizer)
        
        return self._tokenizer
    
    def __call__(self, data: DataDict) -> DataDict:
        # Only tokenize if actions are present (training time)
        if "actions" not in data:
            return data
        
        actions = data["actions"]  # [H, D] - may be per-timestamp normalized
        
        # Extract encoded dimensions
        encoded_chunks = []
        for start, end in self.encoded_dim_ranges:
            encoded_chunks.append(actions[:, start:end])
        encoded_actions = np.concatenate(encoded_chunks, axis=-1)  # [H, D_encoded]
        
        # Convert to global QUANTILE normalization for FAST (always, regardless of per_timestamp)
        # This clips outliers and provides consistent [-1, 1] range for DCT compression
        if self.norm_stats is not None:
            action_stats = self.norm_stats['actions']
            
            # Extract stats for encoded dimensions only
            # Build dimension indices from ranges
            encoded_dims = []
            for start, end in self.encoded_dim_ranges:
                encoded_dims.extend(range(start, end))
            encoded_dims = np.array(encoded_dims)
            
            # Denormalize to get back to raw delta actions
            if self.use_per_timestamp:
                # Denormalize from per-timestamp MEAN/STD normalization
                per_ts_mean = action_stats.per_timestamp_mean[:, encoded_dims]  # [H, D_encoded]
                per_ts_std = action_stats.per_timestamp_std[:, encoded_dims]    # [H, D_encoded]
                actions_delta = encoded_actions * per_ts_std + per_ts_mean
            else:
                # Denormalize from global MEAN/STD normalization
                global_mean = action_stats.mean[encoded_dims]  # [D_encoded]
                global_std = action_stats.std[encoded_dims]    # [D_encoded]
                actions_delta = encoded_actions * global_std + global_mean
            
            # Renormalize using GLOBAL QUANTILE normalization (for FAST)
            # This clips to [q01, q99] and maps to [-1, 1]
            q01 = action_stats.q01[encoded_dims]  # [D_encoded]
            q99 = action_stats.q99[encoded_dims]  # [D_encoded]
            actions_delta = np.clip(actions_delta, q01, q99)
            encoded_actions = 2.0 * (actions_delta - q01) / np.maximum(q99 - q01, 1e-6) - 1.0
        
        # Tokenize
        tokenizer = self._get_tokenizer()
        tokens = tokenizer(encoded_actions[None])[0]
        
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        
        # Truncate or pad to max_fast_tokens
        if len(tokens) > self.max_fast_tokens:
            tokens = tokens[:self.max_fast_tokens]
            mask = np.ones(self.max_fast_tokens, dtype=bool)
        else:
            mask = np.concatenate([
                np.ones(len(tokens), dtype=bool),
                np.zeros(self.max_fast_tokens - len(tokens), dtype=bool)
            ])
            tokens = np.pad(tokens, (0, self.max_fast_tokens - len(tokens)), constant_values=0)
        
        data["fast_tokens"] = tokens.astype(np.int32)
        data["fast_token_mask"] = mask
        
        return data

