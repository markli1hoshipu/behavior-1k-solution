#!/bin/bash
# Multi-node training on 2 x 8 H200 via Slurm.
#
#   sbatch scripts/train_slurm.sbatch pi_behavior_b1k_5task_pilot --overwrite
#
# One task per node, each owning that node's 8 GPUs. jax.distributed.initialize()
# (scripts/train.py::init_distributed) joins them into a single 16-device mesh, driven by
# SLURM_NTASKS / SLURM_PROCID / SLURM_NODELIST. Without it, srun --nodes=2 would just run
# two unrelated 8-GPU jobs -- no error, silently wrong training.
#
# Paths must be under /work6: that is the mount both nodes share. /work is the same
# device on node 106 but is not guaranteed present elsewhere.

#SBATCH --job-name=b1k-train
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=240
#SBATCH --output=/work6/markhsp/behavior-1k-solution/logs/slurm-%j.out
#SBATCH --error=/work6/markhsp/behavior-1k-solution/logs/slurm-%j.out

set -euo pipefail

REPO=/work6/markhsp/behavior-1k-solution
cd "$REPO"

if [ $# -lt 1 ]; then
    echo "usage: sbatch $0 <config-name> [train.py args...]" >&2
    exit 2
fi

export PYTHONPATH="$REPO/src:$REPO/openpi/src"
# HF_HOME explicitly on the shared mount. Do not rely on ~/.cache/huggingface: $HOME is
# node-local (/dev/nvme1n1p1), so the symlink set up on node 106 does not exist on 107,
# which would rebuild the ~61 GB Arrow dataset cache into that node's 123 GB of free root.
export HF_HOME=/work6/markhsp/hf_cache
# JAX's compilation cache stays node-local by design (each node compiles its own kernels).
# Nothing should reach the network mid-run: a declared-but-absent video key makes
# LeRobot fall back to fetching from the Hub, which is 318 round-trips per dataset
# construction. Offline turns that silent stall into an immediate error.
export HF_HUB_OFFLINE=1
# Coordinator port for jax.distributed; override if it collides with another job.
export JAX_COORDINATOR_PORT=${JAX_COORDINATOR_PORT:-12355}

echo "=== job $SLURM_JOB_ID on $SLURM_JOB_NODELIST ==="
echo "config: $*"
echo "started: $(date '+%F %T')"

# Build the Arrow cache once, single-process, before any rank touches it. Both ranks
# otherwise race on the same fingerprint -- HF datasets' lock does not hold across hosts
# on a shared filesystem -- and the loser tries to read the winner's `.incomplete/`
# directory, failing with "OSError: Cannot find data file". Idempotent and fast when the
# cache already exists.
echo "--- prebuilding Arrow cache ---"
.venv/bin/python3 -u scripts/prebuild_cache.py "$1"

# --kill-on-bad-exit so a crash on one node tears down the other instead of leaving it
# spinning on a collective that will never complete.
srun --kill-on-bad-exit=1 \
     --nodes="$SLURM_NNODES" \
     --ntasks-per-node=1 \
     .venv/bin/python3 -u scripts/train.py "$@"

echo "finished: $(date '+%F %T')"
