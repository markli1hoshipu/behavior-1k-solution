"""Training configuration for BEHAVIOR-1K challenge.

Reference: https://github.com/Physical-Intelligence/openpi
"""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import os
import pathlib
from typing import Any, Literal, List, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

# Import from OpenPI
import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.shared.download as _download
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.optimizer as _optimizer
import openpi.transforms as _transforms

# Import from B1K custom modules
from b1k.models import pi_behavior_config
from b1k.policies import b1k_policy
from b1k.shared import normalize as _normalize
from b1k.training import weight_loaders
from b1k import transforms as b1k_transforms

ModelType: TypeAlias = _model.ModelType
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.
    """
    # Assets directory. If not provided, the config assets_dirs will be used.
    assets_dir: str | None = None
    # Asset id. If not provided, the repo id will be used.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False
    # If true, will use per-timestamp normalization for actions
    use_per_timestamp_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt (not used for PI_BEHAVIOR).
    prompt_from_task: bool = False

    # Only used for RLDS data loader.
    rlds_data_dir: str | None = None

    # Only used for B1K data loader.
    behavior_dataset_root: str | None = None

    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # Path to the data filter file for DROID dataset
    filter_dict_path: str | None = None

    # Episodes index to use for training 
    episodes_index: List[int] | None = None


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for B1K."""

    default_prompt: str | None = None  # not used; language comes from tasks.jsonl now
    # B1a annotation directory (annotations/task-XXXX/episode_YYYYYYYY.json).
    # Empty string → annotate transform emits skill_id=0, subtask_progress=0.0.
    annotations_dir: str = ""
    # Rich-prompt source (meta/tasks.jsonl from behavior-1k/2026-challenge-demos).
    task_prompts_path: str = ""
    # Must match PiBehaviorConfig.max_lang_tokens.
    max_lang_tokens: int = 128
    episodes_per_task: int = 200
    # Fraction of samples whose skill name is replaced by "unknown" in the prompt.
    # Keeps skill_pred_from_vlm learnable — see TaskIndexToPromptTokens.
    skill_prompt_dropout: float = 0.5
    skill_dropout_seed: int = 0

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        # Prefer the config-supplied path if present (keeps this factory reusable across configs).
        prompts_path = self.task_prompts_path or getattr(model_config, "task_prompts_path", "")
        lang_len = self.max_lang_tokens or getattr(model_config, "max_lang_tokens", 128)
        return _transforms.Group(
            inputs=[
                _transforms.ResizeImages(224, 224),
                b1k_transforms.AnnotateSkillFromAnnotation(
                    annotations_dir=self.annotations_dir,
                    episodes_per_task=self.episodes_per_task,
                ),
                b1k_transforms.TaskIndexToPromptTokens(
                    task_prompts_path=prompts_path,
                    max_lang_tokens=lang_len,
                    skill_prompt_dropout=self.skill_prompt_dropout,
                    skill_dropout_seed=self.skill_dropout_seed,
                ),
                _transforms.PadStatesAndActions(model_config.action_dim),
            ],
        )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=False,  # Always use z-score normalization for B1K
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class LeRobotB1KDataConfig(DataConfigFactory):
    """Data configuration for BEHAVIOR-1K dataset."""

    action_sequence_keys: Sequence[str] = ("action",)
    use_delta_joint_actions: bool = False

    # FAST auxiliary tokenization (only for PI_BEHAVIOR with use_fast_auxiliary)
    use_fast_tokenization: bool = False

    # B1a / Path B language wiring — passed through to ModelTransformFactory.
    annotations_dir: str = ""
    task_prompts_path: str = ""
    episodes_per_task: int = 200
    skill_prompt_dropout: float = 0.5
    skill_dropout_seed: int = 0

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Repack transforms for B1K observations
        # Camera keys are the official v3.0 names. The old
        # observation.images.rgb.{head,left_wrist,right_wrist} keys came from our own
        # v2.1 conversion and raise KeyError here. Roles per ROBOT_CAMERA_NAMES["R1Pro"]
        # in omnigibson: zed_link = head, left/right_realsense_link = wrists.
        repack_mapping = {
            "observation/egocentric_camera": "observation.rgb.zed_link_camera_0",
            "observation/wrist_image_left": "observation.rgb.left_realsense_link_camera_0",
            "observation/wrist_image_right": "observation.rgb.right_realsense_link_camera_0",
            "observation/state": "observation.state",
            "actions": "action",
            "task_index": "task_index",  # Always preserve task_index
            "timestamp": "timestamp",    # Preserve timestamp for subtask state computation
            "episode_index": "episode_index",  # Preserve episode_index for episode length lookup
            "index": "index",           # Preserve index
        }
            
        repack_transform = _transforms.Group(
            inputs=[_transforms.RepackTransform(repack_mapping)]
        )

        # Prepare data for policy training
        data_transforms = _transforms.Group(
            inputs=[b1k_policy.B1kInputs(model_type=model_config.model_type)],
            outputs=[b1k_policy.B1kOutputs()],
        )

        # Delta action transforms
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(-3, 3, -1, 7, -1, 7, -1)
        else:
            delta_action_mask = _transforms.make_bool_mask(-23)
        
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        # Model transforms (skill+progress annotation, prompt tokenization, padding)
        model_transforms = ModelTransformFactory(
            annotations_dir=self.annotations_dir,
            task_prompts_path=self.task_prompts_path,
            max_lang_tokens=getattr(model_config, "max_lang_tokens", 128),
            episodes_per_task=self.episodes_per_task,
            skill_prompt_dropout=self.skill_prompt_dropout,
            skill_dropout_seed=self.skill_dropout_seed,
        )(model_config)
        
        # FAST tokenization (if enabled for PI_BEHAVIOR)
        if self.use_fast_tokenization and hasattr(model_config, 'use_fast_auxiliary') and model_config.use_fast_auxiliary:
            asset_id = self.assets.asset_id or self.repo_id
            tokenizer_path = assets_dirs / asset_id / "fast_tokenizer"
            
            # Get base config to access norm_stats
            base_config = self.create_base_config(assets_dirs, model_config)
            
            # Only add transform if tokenizer directory exists
            if tokenizer_path.exists():
                model_transforms = model_transforms.push(
                    inputs=[b1k_transforms.TokenizeFASTActions(
                        tokenizer_path=str(tokenizer_path),
                        encoded_dim_ranges=model_config.get_fast_dim_ranges(),
                        max_fast_tokens=model_config.max_fast_tokens,
                        norm_stats=base_config.norm_stats,
                        use_per_timestamp=base_config.use_per_timestamp_norm,
                    )],
                )
            else:
                logging.warning(
                    f"FAST tokenizer not found at {tokenizer_path}. "
                    "FAST auxiliary training will be disabled (inference mode)."
                )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "B1K"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config (PI_BEHAVIOR only for B1K).
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi_behavior_config.PiBehaviorConfig)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Note: PyTorch support removed - JAX only

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=LeRobotB1KDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int | None = None
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader.
    num_workers: int = 2
    # >0 enables ChunkStreamingSampler with this run length (set it to the video GOP).
    # 0 = independent random frames. Only worth it on high-GOP data: the 224 rebuild keeps
    # GOP 250, where a random frame costs ~125 decodes. Correlates samples within a batch.
    chunk_stream_size: int = 0
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # FSDP configuration for model sharding across devices.
    fsdp_devices: int = 1
    
    # Validation configuration
    val_log_interval: int = 100
    val_batch_size: int | None = None
    val_num_batches: int = 10
    val_repo_id: str | None = None
    val_episodes_index: List[int] | None = None
    
    # Number of flow matching samples per training step
    num_flow_samples: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# B1K Training Configurations
_CONFIGS = [
    TrainConfig(
        name="pi_behavior_b1k_fast",
        exp_name="openpi",
        project_name="B1K",
        model=pi_behavior_config.PiBehaviorConfig(
            action_horizon=30,
            action_dim=32,
            use_correlated_noise=True,
            correlation_beta=0.5,
            # FAST auxiliary training
            use_fast_auxiliary=True,
            fast_loss_weight=0.05,
            fast_encoded_dims="0:6,7:23",  # Encode 22 dimensions
            fast_vocab_size=1024,
            max_fast_tokens=200,
            use_kv_transform=True,
            use_knowledge_insulation=False,
            subtask_loss_weight=0.1,
            freeze_vision_backbone=True,
        ),
        data=LeRobotB1KDataConfig(
            repo_id="IliaLarchenko/behavior_224_rgb",
            base_config=DataConfig(
                prompt_from_task=False,  # No text prompts for PI_BEHAVIOR
                behavior_dataset_root="~/data/behavior_224_rgb",
                use_per_timestamp_norm=True,  # Enable per-timestamp normalization
            ),
            use_delta_joint_actions=True,
            use_fast_tokenization=True,  # Enable FAST tokenization in data pipeline
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=1e-4,
            decay_steps=20_000,
            decay_lr=1e-5,
        ),
        num_flow_samples=15,
        weight_loader=weight_loaders.PiBehaviorWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=200_000,
        assets_base_dir="./outputs/assets",
        checkpoint_base_dir="./outputs/checkpoints",
        num_workers=80,
        save_interval=500,
        keep_period=2000,
    ),
    # SMOKE-5TASK: cheap gate on tasks 0, 5, 76, 77 (800 episodes, ~1.6M frames of 211M).
    # Purpose is "does the pipeline run end to end and does loss move", not learning:
    # data load, video decode, 61-D proprio reduction, skill/prompt assembly, FAST
    # tokenization, fwd/bwd, FSDP sharding, checkpoint save. Batch is deliberately small
    # so iterations are fast; real-scale memory is what the 5-task smoke below checks.
    #
    # Task choice spans both halves of the embedding table on purpose: 0 and 5 fall in
    # rows 0..49, which PiBehaviorWeightLoader fills from Ilia's 50-task checkpoint, while
    # 76 and 77 land in the randomly-initialised rows 50..99. A smoke test on warm-started
    # tasks only would not exercise the fresh-embedding path at all.
    #
    # Needs its own assets dir (norm_stats + fast_tokenizer) under
    # ./outputs/assets/pi_behavior_b1k_100_smoke_5task/behavior-1k/2026-challenge-demos.
    TrainConfig(
        name="pi_behavior_b1k_100_smoke_5task",
        exp_name="smoke5",
        project_name="B1K",
        model=pi_behavior_config.PiBehaviorConfig(
            action_horizon=30,
            action_dim=32,
            num_tasks=100,
            max_lang_tokens=128,
            use_correlated_noise=True,
            correlation_beta=0.5,
            use_fast_auxiliary=True,
            fast_loss_weight=0.05,
            fast_encoded_dims="0:6,7:23",
            fast_vocab_size=1024,
            max_fast_tokens=200,
            use_kv_transform=True,
            use_knowledge_insulation=False,
            subtask_loss_weight=0.1,
            freeze_vision_backbone=True,
        ),
        data=LeRobotB1KDataConfig(
            repo_id="behavior-1k/2026-challenge-demos",
            base_config=DataConfig(
                prompt_from_task=False,
                # RGB-only hardlinked view of the official set. Identical content, but
                # info.json drops the depth_linear features we never downloaded --
                # LeRobotDataset walks meta.video_keys checking each file exists, and a
                # declared-but-absent depth video makes it fall back to fetching from the
                # Hub on every construction. Swap to 2026-challenge-demos-224 once the
                # rebuild lands.
                behavior_dataset_root="/work6/markhsp/datasets/2026-challenge-demos-rgbonly",
                use_per_timestamp_norm=True,
                # Task T owns episodes T*200 .. T*200+199. PyArrow predicate pushdown
                # means only the matching rows are materialised, so the Arrow cache is
                # a couple of GB rather than the ~61 GB a full-set load builds.
                episodes_index=(
                    list(range(0, 200))          # task 0  turning_on_radio      (warm-start)
                    + list(range(1000, 1200))    # task 5  setting_mousetraps    (warm-start)
                    + list(range(8000, 8200))    # task 40 make_microwave_popcorn(warm-start)
                    + list(range(15200, 15400))  # task 76 dispose_of_glass      (random init)
                    + list(range(15400, 15600))  # task 77 installing_a_modem    (random init)
                ),
            ),
            use_delta_joint_actions=True,
            use_fast_tokenization=True,
            annotations_dir="/work6/markhsp/datasets/behavior-1k-2026-challenge-demos/annotations",
            task_prompts_path="/work6/markhsp/datasets/behavior-1k-2026-challenge-demos/meta/tasks.jsonl",
            episodes_per_task=200,
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10,
            peak_lr=1e-4,
            decay_steps=100,
            decay_lr=1e-5,
        ),
        num_flow_samples=15,
        weight_loader=weight_loaders.PiBehaviorWeightLoader(
            "/work6/markhsp/models/behavior_50t_checkpoint/params"
        ),
        num_train_steps=100,
        # Default log_interval is 100, which with num_train_steps=100 (steps 0..99) logs
        # exactly once, at step 0 — enough to prove the losses are wired up, useless for
        # confirming they go down. 10 gives a trend without flooding the log.
        log_interval=10,
        assets_base_dir="./outputs/assets",
        checkpoint_base_dir="./outputs/checkpoints",
        num_workers=16,
        batch_size=256,            # 32/GPU across 8 — fast iterations, not real scale
        save_interval=50,          # exercises checkpoint write within the run
        keep_period=100,
        fsdp_devices=8,
        wandb_enabled=False,
    ),
    # PILOT-5TASK: real training run on tasks 0, 5, 40, 76, 77 (1000 episodes, ~6.0M frames).
    # Purpose is "does it actually learn", not a deployable model -- 50k steps at batch 256
    # is ~2.7 epochs over the subset. Watch the three losses separately, not just the total:
    #   action_loss   falling  -> the flow-matching objective is learning
    #   subtask_loss  falling from ~4.6 (== ln 34) -> skill_pred_from_vlm is genuinely
    #                 predicting the skill, i.e. the 50% prompt dropout kept it honest
    #                 rather than letting it read the answer out of the text
    #   fast_loss     falling  -> the retrained FAST tokenizer matches the auxiliary head
    #
    # Do NOT seed the 100-task run from this checkpoint: training on 4 tasks overfits the
    # shared backbone to them, and the other 96 task_embedding rows receive no gradient.
    TrainConfig(
        name="pi_behavior_b1k_5task_pilot",
        exp_name="pilot",
        project_name="B1K",
        model=pi_behavior_config.PiBehaviorConfig(
            action_horizon=30,
            action_dim=32,
            num_tasks=100,
            max_lang_tokens=128,
            use_correlated_noise=True,
            correlation_beta=0.5,
            use_fast_auxiliary=True,
            fast_loss_weight=0.05,
            fast_encoded_dims="0:6,7:23",
            fast_vocab_size=1024,
            max_fast_tokens=200,
            use_kv_transform=True,
            use_knowledge_insulation=False,
            subtask_loss_weight=0.1,
            freeze_vision_backbone=True,
        ),
        data=LeRobotB1KDataConfig(
            repo_id="behavior-1k/2026-challenge-demos",
            base_config=DataConfig(
                prompt_from_task=False,
                # RGB-only hardlinked view; switch to 2026-challenge-demos-224 once that
                # rebuild finishes (its videos are still being written).
                behavior_dataset_root="/work6/markhsp/datasets/2026-challenge-demos-rgbonly",
                use_per_timestamp_norm=True,
                episodes_index=(
                    list(range(0, 200))          # task 0  turning_on_radio      (warm-start)
                    + list(range(1000, 1200))    # task 5  setting_mousetraps    (warm-start)
                    + list(range(8000, 8200))    # task 40 make_microwave_popcorn(warm-start)
                    + list(range(15200, 15400))  # task 76 dispose_of_glass      (random init)
                    + list(range(15400, 15600))  # task 77 installing_a_modem    (random init)
                ),
            ),
            use_delta_joint_actions=True,
            use_fast_tokenization=True,
            annotations_dir="/work6/markhsp/datasets/behavior-1k-2026-challenge-demos/annotations",
            task_prompts_path="/work6/markhsp/datasets/behavior-1k-2026-challenge-demos/meta/tasks.jsonl",
            episodes_per_task=200,
        ),
        # Ilia's peak LR, decayed across the full run rather than their 20k-then-flat
        # (that shape was tied to a 200k-step schedule).
        # 1k warmup, cosine down to 1e-5 by step 10k, then flat 1e-5 for the back half.
        # optax's cosine schedule clamps its step count, so decay_steps < num_train_steps
        # holds at decay_lr rather than continuing the curve — verified at 12k/15k/20k.
        # Same shape as Ilia's (decay early, then constant), scaled to a 20k run.
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=1e-4,
            decay_steps=10_000,
            decay_lr=1e-5,
        ),
        num_flow_samples=15,
        weight_loader=weight_loaders.PiBehaviorWeightLoader(
            "/work6/markhsp/models/behavior_50t_checkpoint/params"
        ),
        # 20k x 1024 = 20.5M samples = ~4.3 epochs over the 4.79M-frame subset. Step count
        # and batch have to be chosen together: the same 20k steps at batch 2048 would be
        # ~8.6 epochs on four tasks, which measures memorisation rather than learning.
        num_train_steps=20_000,
        log_interval=10,
        assets_base_dir="./outputs/assets",
        checkpoint_base_dir="./outputs/checkpoints",
        # Measured on 16 devices (batch / workers -> s/step, throughput, GPU SM):
        #   1024 / 80 -> 5.0 s, 205 samples/s, 79% SM, 35.8 GB peak
        #   2048 / 32 -> 11.6 s, 177 samples/s, 72-79% SM, 79.5 GB peak
        # Both are GPU-bound rather than starved, so worker count buys less than a naive
        # per-worker extrapolation suggests: predicting 25 s/step for 2048/32 from the
        # 1024/80 figures was off by more than 2x, because in the 1024 case the workers
        # were partly idle waiting on the GPUs. 64 sits between the two measured points.
        num_workers=64,
        # 1024 over 8 devices = 128/GPU, the footprint measured at 79.5 GB of 143.8.
        # 2048 on 8 devices would be 256/GPU (~160 GB) and OOM -- that only fit when the
        # mesh spanned 16 devices. Both A/B arms must match here or the comparison is
        # confounded.
        batch_size=1024,
        # A save is ~43 GB and stalls the step loop for minutes. Note save_interval does
        # not divide keep_period, so the only steps both saved and retained are multiples
        # of 10000; a crash rolls back to the last multiple of 2000 (cost 1430 steps once).
        save_interval=2000,
        keep_period=5_000,
        # 8, not 16: the two A/B arms run as separate single-node jobs, one per node, so
        # each sees only its own 8 GPUs. Launch with
        #   sbatch --nodes=1 --nodelist=gpu-h200-106 scripts/train_slurm.sbatch <config> ...
        # init_distributed() returns early when SLURM_NTASKS == 1, so no multi-host setup
        # is involved.
        fsdp_devices=8,
        wandb_enabled=False,
    ),
    # PILOT-5TASK-NOLANG: the A/B counterpart to pi_behavior_b1k_5task_pilot, identical
    # in every field except max_lang_tokens=0, which drops the natural-language span from
    # the prefix and leaves the task identified only by its task_embeddings row -- i.e.
    # Ilia's 2025 design, which won the challenge without any text input.
    #
    # What the comparison is for: the rich descriptions carry object- and step-level detail
    # ("take the grated cheese, the four pieces of pepperoni ... from their tupperware
    # containers in the refrigerator") that a single 2048-d embedding plausibly cannot hold,
    # but that is an argument, not evidence. Run both arms and read action_loss.
    #
    # Generated by copying the pilot block verbatim so the two cannot drift; keep them in
    # sync by hand if either is edited.
    TrainConfig(
        name="pi_behavior_b1k_5task_pilot_nolang",
        exp_name="pilot_nolang",
        project_name="B1K",
        model=pi_behavior_config.PiBehaviorConfig(
            action_horizon=30,
            action_dim=32,
            num_tasks=100,
            max_lang_tokens=0,   # no-language arm
            use_correlated_noise=True,
            correlation_beta=0.5,
            use_fast_auxiliary=True,
            fast_loss_weight=0.05,
            fast_encoded_dims="0:6,7:23",
            fast_vocab_size=1024,
            max_fast_tokens=200,
            use_kv_transform=True,
            use_knowledge_insulation=False,
            subtask_loss_weight=0.1,
            freeze_vision_backbone=True,
        ),
        data=LeRobotB1KDataConfig(
            repo_id="behavior-1k/2026-challenge-demos",
            base_config=DataConfig(
                prompt_from_task=False,
                # RGB-only hardlinked view; switch to 2026-challenge-demos-224 once that
                # rebuild finishes (its videos are still being written).
                behavior_dataset_root="/work6/markhsp/datasets/2026-challenge-demos-rgbonly",
                use_per_timestamp_norm=True,
                episodes_index=(
                    list(range(0, 200))          # task 0  turning_on_radio      (warm-start)
                    + list(range(1000, 1200))    # task 5  setting_mousetraps    (warm-start)
                    + list(range(8000, 8200))    # task 40 make_microwave_popcorn(warm-start)
                    + list(range(15200, 15400))  # task 76 dispose_of_glass      (random init)
                    + list(range(15400, 15600))  # task 77 installing_a_modem    (random init)
                ),
            ),
            use_delta_joint_actions=True,
            use_fast_tokenization=True,
            annotations_dir="/work6/markhsp/datasets/behavior-1k-2026-challenge-demos/annotations",
            task_prompts_path="/work6/markhsp/datasets/behavior-1k-2026-challenge-demos/meta/tasks.jsonl",
            episodes_per_task=200,
        ),
        # Ilia's peak LR, decayed across the full run rather than their 20k-then-flat
        # (that shape was tied to a 200k-step schedule).
        # 1k warmup, cosine down to 1e-5 by step 10k, then flat 1e-5 for the back half.
        # optax's cosine schedule clamps its step count, so decay_steps < num_train_steps
        # holds at decay_lr rather than continuing the curve — verified at 12k/15k/20k.
        # Same shape as Ilia's (decay early, then constant), scaled to a 20k run.
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=1e-4,
            decay_steps=10_000,
            decay_lr=1e-5,
        ),
        num_flow_samples=15,
        weight_loader=weight_loaders.PiBehaviorWeightLoader(
            "/work6/markhsp/models/behavior_50t_checkpoint/params"
        ),
        # 20k x 1024 = 20.5M samples = ~4.3 epochs over the 4.79M-frame subset. Step count
        # and batch have to be chosen together: the same 20k steps at batch 2048 would be
        # ~8.6 epochs on four tasks, which measures memorisation rather than learning.
        # Extended 20k -> 30k. The LR schedule already flattens at decay_lr from step 10k
        # (optax clamps the cosine's step count), so the whole added segment runs at 1e-5
        # without touching lr_schedule.
        num_train_steps=30_000,
        log_interval=10,
        assets_base_dir="./outputs/assets",
        checkpoint_base_dir="./outputs/checkpoints",
        # Measured on 16 devices (batch / workers -> s/step, throughput, GPU SM):
        #   1024 / 80 -> 5.0 s, 205 samples/s, 79% SM, 35.8 GB peak
        #   2048 / 32 -> 11.6 s, 177 samples/s, 72-79% SM, 79.5 GB peak
        # Both are GPU-bound rather than starved, so worker count buys less than a naive
        # per-worker extrapolation suggests: predicting 25 s/step for 2048/32 from the
        # 1024/80 figures was off by more than 2x, because in the 1024 case the workers
        # were partly idle waiting on the GPUs. 64 sits between the two measured points.
        num_workers=64,
        # 224 + chunk_stream_size=250 was tried for the 20k->30k segment and wedged after
        # 60 steps: the log went fully silent (not even the per-batch torchvision warnings)
        # while the job stayed RUNNING. Reverted to the pipeline that carried steps 0-20k.
        # chunk_stream_size=0,
        # 1024 over 8 devices = 128/GPU, the footprint measured at 79.5 GB of 143.8.
        # 2048 on 8 devices would be 256/GPU (~160 GB) and OOM -- that only fit when the
        # mesh spanned 16 devices. Both A/B arms must match here or the comparison is
        # confounded.
        batch_size=1024,
        # A save is ~43 GB and stalls the step loop for minutes. Note save_interval does
        # not divide keep_period, so the only steps both saved and retained are multiples
        # of 10000; a crash rolls back to the last multiple of 2000 (cost 1430 steps once).
        save_interval=2000,
        keep_period=5_000,
        # 8, not 16: the two A/B arms run as separate single-node jobs, one per node, so
        # each sees only its own 8 GPUs. Launch with
        #   sbatch --nodes=1 --nodelist=gpu-h200-106 scripts/train_slurm.sbatch <config> ...
        # init_distributed() returns early when SLURM_NTASKS == 1, so no multi-host setup
        # is involved.
        fsdp_devices=8,
        wandb_enabled=False,
    ),
    # SMOKE: 500-step run on 5 diverse tasks (mix of 2025 and 2026 halves).
    # Goal: catch bugs in the new prompt/skill pipeline before committing to full-scale.
    # Uses Ilia's 50t norm_stats + FAST tokenizer symlinked into ./outputs/assets.
    TrainConfig(
        name="pi_behavior_b1k_100_smoke",
        exp_name="smoke",
        project_name="B1K",
        model=pi_behavior_config.PiBehaviorConfig(
            action_horizon=30,
            action_dim=32,
            num_tasks=100,
            max_lang_tokens=128,
            use_correlated_noise=True,
            correlation_beta=0.5,
            use_fast_auxiliary=True,
            fast_loss_weight=0.05,
            fast_encoded_dims="0:6,7:23",
            fast_vocab_size=1024,
            max_fast_tokens=200,
            use_kv_transform=True,
            use_knowledge_insulation=False,
            subtask_loss_weight=0.1,
            freeze_vision_backbone=True,
        ),
        data=LeRobotB1KDataConfig(
            repo_id="behavior-1k/2026-challenge-demos",
            base_config=DataConfig(
                prompt_from_task=False,
                behavior_dataset_root="/work6/markhsp/datasets/behavior-1k-2026-challenge-demos",
                use_per_timestamp_norm=True,
                # 5 diverse tasks (0, 50, 51, 60, 80) = 1000 episodes. Episode ids are
                # global and contiguous per task: task T owns T*200 .. T*200+199
                # (verified against meta/episodes for all 100 tasks). lerobot >= 0.4.0
                # resolves them via meta.episodes[global_idx], so non-contiguous filters
                # like this work directly — no index remapping needed.
                episodes_index=(
                    list(range(0, 200))            # task 0
                    + list(range(10000, 10200))    # task 50
                    + list(range(10200, 10400))    # task 51
                    + list(range(12000, 12200))    # task 60
                    + list(range(16000, 16200))    # task 80
                ),
            ),
            use_delta_joint_actions=True,
            use_fast_tokenization=True,
            annotations_dir="/work6/markhsp/datasets/behavior-1k-2026-challenge-demos/annotations",
            task_prompts_path="/work6/markhsp/datasets/behavior-1k-2026-challenge-demos/meta/tasks.jsonl",
            episodes_per_task=200,
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=50,
            peak_lr=1e-4,
            decay_steps=100,
            decay_lr=1e-5,
        ),
        num_flow_samples=15,
        weight_loader=weight_loaders.PiBehaviorWeightLoader(
            "/work6/markhsp/models/behavior_50t_checkpoint/params"
        ),
        num_train_steps=100,       # smoke: catch bugs + verify loss decreases
        assets_base_dir="./outputs/assets",
        checkpoint_base_dir="./outputs/checkpoints",
        num_workers=16,            # multiproc dataloading — subclass survives pickle
        batch_size=2048,           # Ilia's README: 8×H200 FSDP → 256/GPU
        save_interval=50,
        keep_period=100,
        fsdp_devices=8,
        wandb_enabled=False,
    ),
    # 100-task 2026 config: Path B (rich language) + B1a (annotation-based skill stages).
    # Warm-starts from Ilia's 50-task pi_behavior checkpoint (task_embeddings 50→100 rows
    # padded with random init; skill_embeddings & skill_pred_from_vlm are fresh random).
    TrainConfig(
        name="pi_behavior_b1k_100",
        exp_name="openpi",
        project_name="B1K",
        model=pi_behavior_config.PiBehaviorConfig(
            action_horizon=30,
            action_dim=32,
            num_tasks=100,
            max_lang_tokens=128,
            use_correlated_noise=True,
            correlation_beta=0.5,
            use_fast_auxiliary=True,
            fast_loss_weight=0.05,
            fast_encoded_dims="0:6,7:23",
            fast_vocab_size=1024,
            max_fast_tokens=200,
            use_kv_transform=True,
            use_knowledge_insulation=False,
            subtask_loss_weight=0.1,
            freeze_vision_backbone=True,
        ),
        data=LeRobotB1KDataConfig(
            repo_id="behavior-1k/2026-challenge-demos",
            base_config=DataConfig(
                prompt_from_task=False,
                behavior_dataset_root="/work6/markhsp/datasets/behavior-1k-2026-challenge-demos",
                use_per_timestamp_norm=True,
                # No episode filter → full 40M-frame training set (all 100 tasks × 200 eps).
                episodes_index=None,
            ),
            use_delta_joint_actions=True,
            use_fast_tokenization=True,
            annotations_dir="/work6/markhsp/datasets/behavior-1k-2026-challenge-demos/annotations",
            task_prompts_path="/work6/markhsp/datasets/behavior-1k-2026-challenge-demos/meta/tasks.jsonl",
            episodes_per_task=200,
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=1e-4,
            decay_steps=20_000,
            decay_lr=1e-5,
        ),
        num_flow_samples=15,
        # Warm-start from Ilia's 50-task intermediate; loader expands task_embeddings
        # [50, 2048] → [100, 2048] (top 50 from ckpt, bottom 50 random), drops the old
        # per-task-stage tables (replaced by B1a global skill_embeddings).
        weight_loader=weight_loaders.PiBehaviorWeightLoader(
            "/work6/markhsp/models/behavior_50t_checkpoint/params"
        ),
        num_train_steps=200_000,
        assets_base_dir="./outputs/assets",
        checkpoint_base_dir="./outputs/checkpoints",
        # NOTE: Ilia's num_workers=80 was tuned for their BehaviorLeRobotDataset with
        # chunk_streaming_using_keyframe. Standard LeRobotDataset + HF datasets doesn't
        # handle that — workers race on the arrow cache and RSS explodes to 200+GB with
        # no forward progress. num_workers=0 confirmed to reach the first train step;
        # bump back up to 4-8 only after validating the pipeline.
        num_workers=0,
        batch_size=2048,           # Ilia's README: 8×H200 FSDP → 256/GPU
        fsdp_devices=8,
        save_interval=500,
        keep_period=2000,
        wandb_enabled=False,       # flip to True after `wandb login`
    ),
]


# ---------------------------------------------------------------------------
# SINGLE-TASK FINETUNES
#
# One specialist per task, to get a per-task ceiling quickly instead of waiting on a
# 100-task run. Every task here is >= 50, i.e. new in 2026: Ilia's 50-task checkpoint has
# never seen any of them, so task_embeddings row N starts random in all twelve.
#
# Sizing: a task is 200 episodes ~= 1.09M frames, so 20k x 256 = 5.12M samples ~= 4.7
# epochs -- the same territory as the 5-task pilot's 4.3, which is what keeps this a
# finetune rather than a memorisation test.
#
# Caveats, both deliberate:
#   - LR is unchanged at peak 1e-4 while batch drops 1024 -> 256, so the per-sample step is
#     4x hotter than the pilot's. If grad_norm blows up early, that is why.
#   - Assets are the 5-task pilot's (tasks 0/5/40/76/77); ten of these twelve are outside
#     that sample. The flow-matching path normalises by per-timestamp mean/std and does not
#     clip, so out-of-distribution actions merely get larger z-scores. The FAST auxiliary
#     does clip to [q01, q99] before tokenising, so its targets are degraded for whatever
#     falls outside -- bounded by fast_loss_weight=0.05.
#
# Launch: one job per node, 8 GPUs each, two at a time.
#   sbatch --nodes=1 --ntasks-per-node=1 scripts/train_slurm.sbatch pi_behavior_b1k_task54_ft
# 16 devices was considered and rejected: at batch 256 that is 16 samples/GPU, where the
# per-step FSDP all-gather of a ~3B model no longer amortises, so two 8-GPU jobs finish all
# twelve in roughly two thirds the wall-clock of twelve sequential 16-GPU jobs.
_SINGLE_TASK_FT = {
    54: "putting_away_toys",
    56: "make_rose_centerpieces",
    59: "organizing_art_supplies",
    70: "thawing_frozen_food",
    73: "organizing_school_stuff",
    74: "carrying_out_garden_furniture",
    76: "dispose_of_glass",
    77: "installing_a_modem",
    85: "putting_dirty_dishes_in_sink",
    86: "make_gift_bags_for_baby_showers",
    87: "collecting_aluminum_cans",
    98: "laying_tile_floors",
}


def _single_task_ft(task_index: int, task_name: str) -> TrainConfig:
    """Build the finetune config for one task.

    Generated rather than written out twelve times: the blocks differ only in
    episodes_index, and hand-copying is how the two pilot arms would have drifted.
    Episodes are laid out 200 per task in task order, so task N owns [N*200, (N+1)*200).
    """
    return TrainConfig(
        name=f"pi_behavior_b1k_task{task_index}_ft",
        # task_name is carried into exp_name so the checkpoint directory says which task it
        # is; with twelve of these, "task74_ft" alone is not enough to read at a glance.
        exp_name=f"task{task_index}_{task_name}",
        project_name="B1K",
        # Identical to pi_behavior_b1k_5task_pilot -- the language+skill arm. Keep it that
        # way: these runs are only interpretable against that pilot.
        model=pi_behavior_config.PiBehaviorConfig(
            action_horizon=30,
            action_dim=32,
            num_tasks=100,
            max_lang_tokens=128,
            use_correlated_noise=True,
            correlation_beta=0.5,
            use_fast_auxiliary=True,
            fast_loss_weight=0.05,
            fast_encoded_dims="0:6,7:23",
            fast_vocab_size=1024,
            max_fast_tokens=200,
            use_kv_transform=True,
            use_knowledge_insulation=False,
            subtask_loss_weight=0.1,
            freeze_vision_backbone=True,
        ),
        data=LeRobotB1KDataConfig(
            repo_id="behavior-1k/2026-challenge-demos",
            base_config=DataConfig(
                prompt_from_task=False,
                behavior_dataset_root="/work6/markhsp/datasets/2026-challenge-demos-rgbonly",
                use_per_timestamp_norm=True,
                episodes_index=list(range(task_index * 200, (task_index + 1) * 200)),
            ),
            use_delta_joint_actions=True,
            use_fast_tokenization=True,
            annotations_dir="/work6/markhsp/datasets/behavior-1k-2026-challenge-demos/annotations",
            task_prompts_path="/work6/markhsp/datasets/behavior-1k-2026-challenge-demos/meta/tasks.jsonl",
            episodes_per_task=200,
        ),
        # Same shape as the pilot: 1k warmup, cosine to 1e-5 by 10k, flat 1e-5 to 20k.
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=1e-4,
            decay_steps=10_000,
            decay_lr=1e-5,
        ),
        num_flow_samples=15,
        weight_loader=weight_loaders.PiBehaviorWeightLoader(
            "/work6/markhsp/models/behavior_50t_checkpoint/params"
        ),
        # Cut from 20k to end earlier. 10_001, not 10_000: the loop is
        # range(start_step, num_train_steps), so this makes 10000 the last step, where
        # `step % save_interval == 0` and `step == num_train_steps - 1` coincide. All twelve
        # checkpoints are then named 10000/ and keep_period protects every one of them;
        # num_train_steps=10_000 would have ended at 9999, which is neither a save-interval
        # multiple nor period-protected, so it would survive only as "most recent".
        #
        # decay_steps stays at 10_000 even though the run now ends there, so the LR bottoms
        # out at 1e-5 exactly as the run finishes and there is no constant-LR tail. That is
        # a real loss -- the nolang pilot's second 10k at flat 1e-5 took action_loss from
        # 0.0225 to 0.0188. It is deliberate: tasks 54 and 56 were already running under
        # decay_steps=10_000 and can only be stopped, not rewound, so moving the decay for
        # the other ten would leave the twelve models incomparable.
        num_train_steps=10_001,
        log_interval=10,
        assets_base_dir="./outputs/assets",
        checkpoint_base_dir="./outputs/checkpoints",
        # 256 over 8 devices = 32/GPU, ~1/4 of the pilot's footprint, so nowhere near the
        # 143.8 GB ceiling. Workers stay at 64: at ~0.39 s/sample/worker that supplies
        # ~164 samples/s against a GPU-bound step of ~100 samples/s, so the loader keeps
        # ahead without the RSS of 80.
        batch_size=256,
        num_workers=64,
        fsdp_devices=8,
        # save_interval divides keep_period here, unlike the pilot: saves land on 5000,
        # 10000, 15000 and the final 19999, and 10000 is period-protected. The pilot's
        # 2000/5000 pairing left the final checkpoint alive only as "most recent", which is
        # how pilot_nolang/19999 was deleted when the 30k extension wrote 26000.
        save_interval=5_000,
        keep_period=10_000,
        wandb_enabled=False,
    )


_CONFIGS += [_single_task_ft(idx, name) for idx, name in _SINGLE_TASK_FT.items()]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]

