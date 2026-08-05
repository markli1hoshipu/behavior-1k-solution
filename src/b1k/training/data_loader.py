"""Data loading for BEHAVIOR-1K dataset.

Reference: https://github.com/wensi-ai/openpi/tree/behavior
"""

import logging
import time

import torch

# Import all base data loading from OpenPI
from openpi.training.data_loader import (
    Dataset,
    IterableDataset,
    DataLoader,
    TransformedDataset,
    IterableTransformedDataset,
    FakeDataset,
    TorchDataLoader,
    RLDSDataLoader,
    create_torch_dataset,
    create_rlds_dataset,
    transform_iterable_dataset,
    create_data_loader,
    create_torch_data_loader,
    create_rlds_data_loader,
)

import openpi.training.config as _config
import openpi.transforms as _transforms

from b1k.models.observation import Observation
from b1k.transforms_normalize import NormalizeWithPerTimestamp


from lerobot.datasets.lerobot_dataset import LeRobotDataset as _LeRobotDataset


class B1kLeRobotDataset(_LeRobotDataset):
    """LeRobotDataset with a row-first _query_hf_dataset.

    Upstream 0.4.4 does:

        result[key] = torch.stack(self.hf_dataset[key][relative_indices])

    described in its docstring as "column-first [key][indices] for speed". On a
    `datasets.Dataset` carrying a formatting transform that is backwards and
    catastrophic: indexing by column name materialises the *entire* column through
    hf_transform_to_torch before 30 of its values are selected. Profiled on the 4-task
    smoke subset (4.79M rows), one __getitem__ took 187 s, of which 176 s was
    format_column and 71 s was 4,791,222 torch.tensor calls — one per row in the
    dataset, per sample. Video decoding was 3.1 s of that 187 s, so this, not the codec
    or the GOP, is what made the earlier attempt run for 26 h without completing a step.

    Row-first touches only the ~30 requested rows. Upstream's own except-branch already
    does this, but the column-first path never raises, it just crawls.

    Overriding a private method is deliberate: the alternative is dropping the
    formatting transform, which the rest of LeRobot relies on.
    """

    def _query_hf_dataset(self, query_indices: dict[str, list[int]]) -> dict:
        result: dict = {}
        for key, q_idx in query_indices.items():
            if key in self.meta.video_keys:
                continue
            relative_indices = (
                q_idx
                if self._absolute_to_relative_idx is None
                else [self._absolute_to_relative_idx[idx] for idx in q_idx]
            )
            result[key] = torch.stack(self.hf_dataset[relative_indices][key])
        return result


class DataLoaderImpl(DataLoader):
    """Custom DataLoader using our Observation with fast_tokens."""
    
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield Observation.from_dict(batch), batch["actions"]


def create_behavior_dataset(data_config: _config.DataConfig, action_horizon: int, seed: int | None = None) -> Dataset:
    """Create a BEHAVIOR-1K dataset for training.
    
    Uses OmniGibson's BehaviorLeRobotDataset for efficient loading of BEHAVIOR-1K data.
    
    Args:
        data_config: Data configuration
        action_horizon: Action horizon for delta timestamps
        seed: Random seed for shuffling. If None, uses random seed based on current time.
    
    Returns:
        Dataset instance with BEHAVIOR-1K data
    """
    # Use our subclass that fixes the LeRobot v3 episode-filter bug (defined at module scope
    # so pickle can restore it in DataLoader workers regardless of start-method).
    LeRobotDataset = B1kLeRobotDataset

    if seed is None:
        seed = int(time.time() * 1000) % (2**32)
        logging.info(f"Using random seed for LeRobotDataset: {seed}")

    # Task-name list (kept for reference/logging; no longer used as a filter — episodes
    # are selected via data_config.episodes_index which is more precise.)
    tasks = [
    "picking_up_trash", # difficulty: 2
    "putting_away_Halloween_decorations", # difficulty: 3
    "cleaning_up_plates_and_food", # difficulty: 3.5
    "setting_mousetraps", # difficulty: 2
    "hiding_Easter_eggs", # difficulty: 2
    "set_up_a_coffee_station_in_your_kitchen", # difficulty: 3
    "putting_dishes_away_after_cleaning", # difficulty: 3
    "preparing_lunch_box", # difficulty: 3
    "loading_the_car", # difficulty: 3.5
    "carrying_in_groceries", # difficulty: 3.5
    "turning_on_radio", # difficulty: 1
    "picking_up_toys", # difficulty: 3.5
    "can_meat", # difficulty: 3.5
    "rearranging_kitchen_furniture", # difficulty: 3
    "putting_up_Christmas_decorations_inside", # difficulty: 2
    "bringing_in_wood", # difficulty: 1.5
    "moving_boxes_to_storage", # difficulty: 1.5
    "bringing_water", # difficulty: 1.5
    "tidying_bedroom", # difficulty: 2
    "outfit_a_basic_toolbox", # difficulty: 2,
    "sorting_vegetables",
    "collecting_childrens_toys",
    "putting_shoes_on_rack",
    "boxing_books_up_for_storage",
    "storing_food",
    "clearing_food_from_table_into_fridge",
    "assembling_gift_baskets",
    "sorting_household_items",
    "getting_organized_for_work",
    "clean_up_your_desk",
    "setting_the_fire",
    "clean_boxing_gloves",
    "wash_a_baseball_cap",
    "wash_dog_toys",
    "hanging_pictures",
    "attach_a_camera_to_a_tripod",
    "clean_a_patio",
    "clean_a_trumpet",
    "spraying_for_bugs",
    "spraying_fruit_trees",
    "make_microwave_popcorn",
    "cook_cabbage",
    "make_pizza",
    "chop_an_onion",
    "slicing_vegetables",
    "chopping_wood",
    "canning_food",
    "cook_hot_dogs",
    "cook_bacon",
    "freeze_pies",
    ]
    
    dataset = LeRobotDataset(
        repo_id=data_config.repo_id,
        root=data_config.behavior_dataset_root,
        episodes=data_config.episodes_index,
        delta_timestamps={
            key: [t / 30.0 for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
        # Standard LeRobot loader — leaves task-name filtering / annotation loading to
        # our downstream transforms (task_index is present in the parquet).
        tolerance_s=0.1,
        video_backend="pyav",  # pyav is more forgiving on ffmpeg shared lib version mismatches
    )

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset.meta.tasks)])

    return dataset


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform dataset with B1K-specific per-timestamp normalization support.
    
    CRITICAL: This overrides wensi-ai's transform_dataset to pass use_per_timestamp to Normalize.
    wensi-ai's version doesn't support per-timestamp normalization which causes huge action losses!
    """
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    # Build transform list
    transforms_list = [
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
        # Use custom Normalize with per-timestamp support (wensi-ai's doesn't have it!)
        NormalizeWithPerTimestamp(
            norm_stats, 
            use_quantiles=data_config.use_quantile_norm,
            use_per_timestamp=data_config.use_per_timestamp_norm  # CRITICAL: Per-timestamp normalization!
        ),
    ]
    
    # B1a: annotation-based skill labels + progress are computed by
    # AnnotateSkillFromAnnotation, which reads per-episode annotation JSONs directly
    # from the annotations_dir set in the DataConfig — no dataset reference needed.
    # (The old ComputeSubtaskStateFromMeta injection is retired.)
    transforms_list.extend(data_config.model_transforms.inputs)

    return TransformedDataset(dataset, transforms_list)


def create_behavior_data_loader(
    config: _config.TrainConfig,
    *,
    sharding=None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
) -> DataLoader:
    """Create a data loader for BEHAVIOR-1K training."""
    import jax
    import time
    
    data_config = config.data.create(config.assets_dirs, config.model)
    
    # Use random seed if not provided
    seed = config.seed
    if seed is None:
        seed = int(time.time() * 1000) % (2**32)
        logging.info(f"Using random seed: {seed}")
    
    dataset = create_behavior_dataset(data_config, action_horizon=config.model.action_horizon, seed=seed)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=config.batch_size // jax.process_count(),
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=seed,
    )
    
    return DataLoaderImpl(data_config, data_loader)
