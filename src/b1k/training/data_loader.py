"""Data loading for BEHAVIOR-1K dataset.

Reference: https://github.com/wensi-ai/openpi/tree/behavior
"""

import logging
import multiprocessing
import time

import jax
import numpy as np
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

from openpi.training.data_loader import _collate_fn as _openpi_collate_fn
from openpi.training.data_loader import _worker_init_fn as _openpi_worker_init_fn

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


class ChunkStreamingSampler(torch.utils.data.Sampler):
    """Sample contiguous runs instead of independent random frames.

    Why: JackLiu's 224 rebuild keeps the source GOP of 250, so a random frame costs ~125
    frame-decodes (start at the preceding keyframe, decode forward). Drawing a *run* of
    consecutive frames amortises that to roughly one decode per sample, which is the whole
    point of Ilia's chunk_streaming_using_keyframe -- reimplemented here as a sampler
    because their BehaviorLeRobotDataset only reads the 2025 v2.1 layout (one episode per
    file, task id derived as episode_index // 1e4, a hardcoded 50-task table) and upstream
    has since deleted it.

    Runs never cross an episode boundary: consecutive rows of two different demonstrations
    are unrelated, and the action-horizon window would silently blend them.

    This is a change in training semantics, not just I/O. A batch now holds a few long runs
    rather than batch_size independent draws, so samples within it are strongly correlated
    and the effective batch is much smaller than the nominal one. Ilia ran batch 2048 with
    this scheme; using it at a smaller batch trades gradient quality for throughput.
    """

    def __init__(self, episode_index, chunk_size: int = 250, seed: int = 0, drop_last: bool = True):
        ep = np.asarray(episode_index)
        # The filtered dataset keeps rows in order, so each episode occupies one contiguous
        # span; boundaries are exactly where episode_index changes.
        bounds = np.flatnonzero(np.diff(ep)) + 1
        starts = np.concatenate(([0], bounds))
        ends = np.concatenate((bounds, [len(ep)]))

        self.chunks = []
        for a, b in zip(starts, ends):
            for c in range(a, b, chunk_size):
                d = min(c + chunk_size, b)
                if d > c and (not drop_last or d - c == chunk_size):
                    self.chunks.append((c, d))
        self.chunk_size = chunk_size
        self.seed = seed
        self.epoch = 0
        self._len = sum(b - a for a, b in self.chunks)
        logging.info(
            f"ChunkStreamingSampler: {len(self.chunks)} chunks of {chunk_size} over "
            f"{len(starts)} episodes, {self._len} of {len(ep)} frames reachable"
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self._len

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        for i in rng.permutation(len(self.chunks)):
            a, b = self.chunks[i]
            yield from range(a, b)


class _MultiHostTorchDataLoader:
    """Torch loader for multi-host JAX, with each rank reading a disjoint shard.

    openpi finished multi-host loading only for the torch.distributed path: its JAX branch
    divides the batch size by process_count but never shards the *indices*, so
    TorchDataLoader.__init__ refuses to run at all:

        NotImplementedError("Data loading with multiple processes is not supported.")

    Rather than defeat that guard (patching jax.process_count globally would also corrupt
    checkpoints.py, which branches on jax.process_index()), this builds the torch loader
    directly. It is deliberately a near-copy of TorchDataLoader with two differences: a
    DistributedSampler supplies the per-rank shard, and there is no guard. openpi's own
    collate and worker-init are reused so batch layout cannot drift from the single-host
    path.

    Correctness rests on the sampler: jax.make_array_from_process_local_data assembles the
    global array from each rank's slice, so if the ranks drew overlapping samples the global
    batch would silently contain duplicates -- a smaller effective batch and correlated
    gradients, with no error.
    """

    def __init__(
        self,
        dataset,
        *,
        local_batch_size: int,
        sharding,
        sampler,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
    ):
        if len(dataset) < local_batch_size:
            raise ValueError(
                f"Local batch size ({local_batch_size}) is larger than the dataset "
                f"size ({len(dataset)})."
            )
        self._sharding = sharding
        self._num_batches = num_batches
        self._sampler = sampler

        mp_context = multiprocessing.get_context("spawn") if num_workers > 0 else None
        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=local_batch_size,
            shuffle=False,          # the sampler owns ordering
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_openpi_collate_fn,
            worker_init_fn=_openpi_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    def __iter__(self):
        num_items = 0
        epoch = 0
        while True:
            # DistributedSampler reseeds from its epoch; without this every pass over the
            # shard would repeat the same order.
            self._sampler.set_epoch(epoch)
            epoch += 1
            for batch in self._data_loader:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                num_items += 1
                yield jax.tree.map(
                    lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch
                )


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
    
    raw = create_behavior_dataset(data_config, action_horizon=config.model.action_horizon, seed=seed)

    # Chunk streaming needs the episode layout, read straight off the Arrow column. Going
    # through hf_dataset["episode_index"] would run the formatting transform over every row
    # -- the same trap that made __getitem__ take 187 s (see B1kLeRobotDataset).
    chunk_sampler = None
    chunk_size = getattr(config, "chunk_stream_size", 0) or 0
    if chunk_size > 0:
        base = raw.dataset if isinstance(raw, TransformedDataset) else raw
        ep_col = base.hf_dataset.data.column("episode_index").to_numpy(zero_copy_only=False)
        chunk_sampler = ChunkStreamingSampler(ep_col, chunk_size=chunk_size, seed=seed)

    dataset = transform_dataset(raw, data_config, skip_norm_stats=skip_norm_stats)

    # Multi-host data loading. openpi finished this only for the torch.distributed path:
    # its JAX branch divides the batch size by process_count but never shards the *indices*,
    # so TorchDataLoader.__init__ guards with
    #     NotImplementedError("Data loading with multiple processes is not supported.")
    # The rest of the machinery is already there -- __iter__ assembles the global array with
    # jax.make_array_from_process_local_data, which is exactly the multi-host API.
    #
    # What was missing is that each rank must see a *disjoint* shard. Without that, both
    # ranks draw from the whole dataset and the assembled global batch contains duplicated
    # samples: no error, just a smaller effective batch and correlated gradients.
    # DistributedSampler gives disjoint shards from a shared seed. TorchDataLoader already
    # accepts a sampler and disables its own shuffle when one is passed.
    n_proc = jax.process_count()
    local_batch_size = config.batch_size // n_proc

    if n_proc == 1:
        return DataLoaderImpl(
            data_config,
            TorchDataLoader(
                dataset,
                local_batch_size=local_batch_size,
                sharding=sharding,
                shuffle=shuffle and chunk_sampler is None,
                sampler=chunk_sampler,
                num_batches=num_batches,
                num_workers=config.num_workers,
                seed=seed,
            ),
        )

    if chunk_sampler is not None:
        raise NotImplementedError(
            "chunk_stream_size with multi-host is not implemented: each rank would need a "
            "disjoint slice of the *chunk* list, not of the frame list, or the ranks would "
            "stream overlapping runs. Run chunk streaming single-node for now."
        )
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=n_proc,
        rank=jax.process_index(),
        shuffle=shuffle,
        seed=seed,
        drop_last=True,
    )
    logging.info(
        f"multi-host data loading: rank {jax.process_index()}/{n_proc}, "
        f"{len(sampler)} of {len(dataset)} samples, local_batch_size={local_batch_size}"
    )
    return DataLoaderImpl(
        data_config,
        _MultiHostTorchDataLoader(
            dataset,
            local_batch_size=local_batch_size,
            sharding=sharding,
            sampler=sampler,
            num_batches=num_batches,
            num_workers=config.num_workers,
            seed=seed,
        ),
    )
