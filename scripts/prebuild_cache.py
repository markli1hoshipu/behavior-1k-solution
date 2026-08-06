"""Build the HF datasets Arrow cache for a config in a single process, then exit.

Multi-host launches must not build this cache concurrently. Both ranks call
load_nested_dataset() with the same fingerprint, HF datasets' lock does not hold across
hosts on a shared filesystem, and the loser reads from the winner's
`<hash>.incomplete/` directory:

    OSError: Cannot find data file.
    [Errno 2] No such file or directory: '.../<hash>.incomplete/parquet-train-...arrow'

Running this once before srun leaves a complete cache that every rank simply
memory-maps. Idempotent: with the cache already present it returns in seconds.

    python scripts/prebuild_cache.py pi_behavior_b1k_5task_pilot
"""
import sys
import time

import b1k.training.config as _config
from b1k.training import data_loader as _dl


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <config-name>", file=sys.stderr)
        return 2
    name = sys.argv[1]
    config = _config.get_config(name)
    data_config = config.data.create(config.assets_dirs, config.model)

    print(f"prebuilding Arrow cache for {name}")
    print(f"  root     = {data_config.behavior_dataset_root}")
    eps = data_config.episodes_index
    print(f"  episodes = {len(eps) if eps else 'all'}")

    t0 = time.perf_counter()
    ds = _dl.create_behavior_dataset(
        data_config, action_horizon=config.model.action_horizon, seed=0
    )
    print(f"  frames   = {len(ds)}")
    print(f"  done in {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
