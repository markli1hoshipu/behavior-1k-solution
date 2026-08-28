"""Chunk-aware inference server for the b1k solution (Hoshipu pi05-b1k5t / 100t
checkpoints). Drop-in replacement for `scripts/serve_b1k.py` that speaks the
`need_obs` / `reuse_cached_action` wire protocol expected by
`omnigibson.eval.eval_fast` from /shared_work/BEHAVIOR-2026.

Protocol
--------
Client (`omnigibson.eval.eval_fast.ChunkAwareWebsocketClientPolicy`):
  - Connects, receives initial `{"model": <str>, "replan_steps": <int>}` metadata.
  - Per env step:
      • sends the full obs dict on `need_obs` steps (msgpack + numpy);
      • sends `{"reuse_cached_action": True}` on cached-chunk steps (no obs).
      • sends `{"reset": True}` when the episode ends (no reply expected).

Server (this file):
  - Receives obs → feeds it into a `B1KPolicyWrapper` identical to the one in
    `serve_b1k.py`. The wrapper owns the chunk logic (30-action infer, 26→20
    cubic interpolation, rolling inpainting, skill voting, correction rules),
    and its `act(obs)` returns one action per env step.
  - On cached steps we still call `wrapper.act(cached_obs)` so its internal
    step_count / subtask_progress / action_index advance correctly; the wire
    win comes from the client skipping the ~1 MB image payload, not from
    server-side compute.

Only the wire protocol differs from `serve_b1k.py`; the policy load path and
`B1KWrapperConfig` are shared verbatim so both servers stay in lockstep.
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import functools
import http
import logging
import os
import pathlib
import socket
import time
import traceback

# JAX memory allocation must be set before importing JAX (mirrors serve_b1k.py).
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import msgpack
import numpy as np
import websockets.asyncio.server as ws_server

from openpi.policies import policy as _policy
from b1k.policies import policy_config as _policy_config
from b1k.policies.checkpoint_switcher import CheckpointSwitcher
from b1k.shared.eval_b1k_wrapper import B1KPolicyWrapper, B1KWrapperConfig
from b1k.training import config as _config


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# msgpack + numpy codec (must match omnigibson.eval.utils.network_utils on the
# client). Torch tensors arrive already numpy-packed by the client, so we only
# need to decode ndarrays here.
# ---------------------------------------------------------------------------
def _pack_array(obj):
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(),
                b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]),
                          shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


Packer = functools.partial(msgpack.Packer, default=_pack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


def _health_check(connection, request):
    """`/healthz` endpoint for the client's pre-connect probe (matches
    `WebsocketClientPolicy._wait_for_server`)."""
    if getattr(request, "path", None) == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


# ---------------------------------------------------------------------------
# Chunk-aware WebSocket server
# ---------------------------------------------------------------------------
class B1KChunkAwareServer:
    def __init__(self, wrapper: B1KPolicyWrapper, host: str, port: int,
                 replan_steps: int):
        self.wrapper = wrapper
        self.host = host
        self.port = port
        # Client uses this many actions per chunk before re-querying with a
        # fresh obs. Matches the wrapper's `execute_in_n_steps`, so
        # `need_obs=True` fires exactly on wrapper re-infer boundaries.
        self.replan_steps = replan_steps

    def serve_forever(self):
        asyncio.run(self._run())

    async def _run(self):
        logger.info(f"b1k chunk-aware server starting on {self.host}:{self.port}")
        async with ws_server.serve(
            self._handler, self.host, self.port,
            compression=None, max_size=None, process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket):
        peer = websocket.remote_address
        logger.info(f"connection opened: {peer}")
        packer = Packer()

        # Initial metadata (client reads .get_server_metadata() from here).
        await websocket.send(packer.pack({
            "model": "b1k-chunkaware",
            "replan_steps": self.replan_steps,
        }))

        # Cache the last full-obs so we can feed it back into wrapper.act()
        # on `reuse_cached_action` steps -- the wrapper needs the raw dict for
        # its per-step `extract_state_from_proprio` and skill-voting logic,
        # even though it will not run policy.infer on cached steps.
        cached_obs: dict | None = None
        try:
            async for raw in websocket:
                t0 = time.monotonic()
                req = unpackb(raw, strict_map_key=False)

                # Client's episode-end reset: no reply expected (matches
                # WebsocketClientPolicy behaviour in the stock harness).
                if isinstance(req, dict) and "reset" in req and len(req) == 1:
                    self.wrapper.reset()
                    cached_obs = None
                    logger.info("policy reset received")
                    continue

                reuse_cached = isinstance(req, dict) and req.get("reuse_cached_action", False)

                if reuse_cached:
                    if cached_obs is None:
                        # Client asked for a cached action before we ever saw a
                        # real obs. Only happens on desync; fail loud so the
                        # eval harness surfaces the bug instead of running with
                        # a stale action buffer.
                        raise RuntimeError(
                            "reuse_cached_action received before any full obs "
                            "(client/server desync on need_obs)"
                        )
                    obs_for_wrapper = cached_obs
                else:
                    obs_for_wrapper = req
                    cached_obs = req

                action = self.wrapper.act(obs_for_wrapper)
                # wrapper.act returns torch.Tensor(23,) float32 already sliced.
                action_np = action.detach().cpu().numpy().astype(np.float32)

                # `need_obs` reports the state *after* this call: the next
                # wrapper.act will trigger a fresh infer iff action_index has
                # wrapped past execute_in_n_steps. Kept consistent with the
                # wrapper's own re-infer condition so the client never sends a
                # `reuse_cached_action` that would surprise it.
                need_obs = (self.wrapper.action_index >=
                            self.wrapper.config.execute_in_n_steps)

                resp = {
                    "action": action_np,
                    "infer_ms": (time.monotonic() - t0) * 1000.0,
                    "need_obs": need_obs,
                }
                await websocket.send(packer.pack(resp))
        except Exception:
            logger.error("handler error:\n%s", traceback.format_exc())
            try:
                await websocket.send(packer.pack({"error": traceback.format_exc()}))
                await websocket.close(code=1011, reason="server error")
            except Exception:
                pass
        finally:
            logger.info(f"connection closed: {peer}")


# ---------------------------------------------------------------------------
# Policy construction — matches serve_b1k.py's main() up to the wrapper build.
# ---------------------------------------------------------------------------
def build_wrapper(config_name: str, ckpt_dir: str,
                  task_id: int | None,
                  wrapper_config: B1KWrapperConfig,
                  task_checkpoint_mapping: str | None) -> B1KPolicyWrapper:
    logger.info(f"Loading b1k policy: config={config_name} ckpt={ckpt_dir}")
    train_config = _config.get_config(config_name)

    sample_kwargs = {"num_steps": wrapper_config.num_steps}
    base_policy = _policy_config.create_trained_policy(
        train_config, ckpt_dir,
        default_prompt=None,
        sample_kwargs=sample_kwargs,
    )

    checkpoint_switcher = None
    if task_checkpoint_mapping:
        logger.info(f"Multi-checkpoint mode: {task_checkpoint_mapping}")
        checkpoint_switcher = CheckpointSwitcher(
            config_path=task_checkpoint_mapping,
            training_config=train_config,
            sample_kwargs=sample_kwargs,
        )

    max_lang_tokens = getattr(train_config.model, "max_lang_tokens", 0)
    task_prompts = {}
    if max_lang_tokens > 0:
        prompts_path = getattr(train_config.data, "task_prompts_path", "") or getattr(
            train_config.model, "task_prompts_path", ""
        )
        if prompts_path:
            from b1k.transforms import _load_task_prompts
            task_prompts = _load_task_prompts(prompts_path)
            logger.info(f"language arm: {len(task_prompts)} task prompts, "
                        f"max_lang_tokens={max_lang_tokens}")
        else:
            logger.warning("max_lang_tokens > 0 but no task_prompts_path; prompts "
                           "will be placeholders and will not match training.")
    else:
        logger.info("no-language arm: prompt is [task_id, skill_id]")

    wrapper = B1KPolicyWrapper(
        base_policy,
        text_prompt="PI_BEHAVIOR model (task-conditioned)",
        task_id=task_id,
        config=wrapper_config,
        checkpoint_switcher=checkpoint_switcher,
        max_lang_tokens=max_lang_tokens,
        task_prompts=task_prompts,
    )
    return wrapper


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="b1k chunk-aware WebSocket inference server")
    ap.add_argument("--config-name", required=True,
                    help="b1k TrainConfig name, e.g. pi_behavior_b1k_5task_pilot_nolang")
    ap.add_argument("--ckpt-dir", required=True,
                    help="Path to orbax checkpoint directory (contains params/ and assets/)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--task-id", type=int, default=None,
                    help="Optional: pin task_id at server start. Otherwise extracted "
                         "from obs['task_id'] each step (default; matches evaluator.py).")
    ap.add_argument("--task-checkpoint-mapping", default=None,
                    help="Optional JSON mapping task_id -> ckpt dir for multi-ckpt mode.")
    # B1KWrapperConfig knobs (defaults match serve_b1k.py).
    ap.add_argument("--actions-to-execute", type=int, default=26)
    ap.add_argument("--actions-to-keep", type=int, default=4)
    ap.add_argument("--execute-in-n-steps", type=int, default=20)
    ap.add_argument("--history-len", type=int, default=3)
    ap.add_argument("--votes-to-promote", type=int, default=2)
    ap.add_argument("--time-threshold-inpaint", type=float, default=0.3)
    ap.add_argument("--num-steps", type=int, default=20)
    ap.add_argument("--disable-eval-tricks", action="store_true",
                    help="Turn off correction rules and gripper-variation compression check.")
    ap.add_argument("--pinned-skill-id", type=int, default=None,
                    help="Freeze current_skill_id at this value (0=unknown sentinel). "
                         "Disables the wrapper's majority-vote skill update AND correction "
                         "rules that would overwrite the skill. Use to ablate the skill "
                         "signal in the prompt.")
    ap.add_argument("--action-log-dir", type=str, default=None,
                    help="If set, per-step actions/predictions get appended to "
                         "<dir>/<task>_<inst>_<rollout>.jsonl. One line per act() call: "
                         "{step, skill_id, subtask_progress, action:[23]}. "
                         "Use with annotate_video.py to overlay predictions on the "
                         "sampled MP4. Opt-in; disabled by default (no perf impact).")
    ap.add_argument("--pinned-task-id", type=int, default=None,
                    help="Freeze wrapper.task_id at this value regardless of obs['task_id']. "
                         "Disables _handle_task_change so obs-driven task switches are ignored. "
                         "Use to ablate task-embedding conditioning (e.g. --pinned-task-id 0 "
                         "forces every rollout to look up task_embeddings[0], even on t45/t46).")
    args = ap.parse_args()

    wrapper_config = B1KWrapperConfig(
        actions_to_execute=args.actions_to_execute,
        actions_to_keep=args.actions_to_keep,
        execute_in_n_steps=args.execute_in_n_steps,
        history_len=args.history_len,
        votes_to_promote=args.votes_to_promote,
        time_threshold_inpaint=args.time_threshold_inpaint,
        num_steps=args.num_steps,
        apply_eval_tricks=not args.disable_eval_tricks,
    )

    wrapper = build_wrapper(
        config_name=args.config_name,
        ckpt_dir=args.ckpt_dir,
        task_id=args.task_id,
        wrapper_config=wrapper_config,
        task_checkpoint_mapping=args.task_checkpoint_mapping,
    )

    # Task-pin ablation: freeze wrapper.task_id at `pinned_tid` no matter what
    # obs['task_id'] says. Neutralises _handle_task_change so the wrapper never
    # switches task, and pre-writes task_id so prepare_batch_for_pi_behavior
    # emits [pinned_tid, skill_id] as the prompt. Use to ablate task-embedding
    # conditioning (does the model still work with a wrong task row?).
    if args.pinned_task_id is not None:
        pinned_tid = int(args.pinned_task_id)
        wrapper.task_id = pinned_tid
        wrapper._handle_task_change = lambda *_a, **_kw: None
        _orig_reset_t = wrapper.reset
        def _reset_task_pinned():
            _orig_reset_t()
            wrapper.task_id = pinned_tid
        wrapper.reset = _reset_task_pinned
        logger.info(f"pinned task_id={pinned_tid} (obs-driven task switches disabled)")

    # Skill-pin ablation: freeze current_skill_id at `pinned` no matter what.
    # Three writers to current_skill_id inside wrapper: (1) reset() clears it
    # back to 0, (2) update_current_stage() promotes on majority vote,
    # (3) apply_correction_rules inside act() overwrites if it disagrees. We
    # neutralise (2) via a no-op, wrap (1) to re-pin after each episode reset,
    # and wrap act() to force-reassign after every call so (3) can't stick.
    if args.pinned_skill_id is not None:
        pinned = int(args.pinned_skill_id)
        wrapper.current_skill_id = pinned
        wrapper.update_current_stage = lambda *_a, **_kw: None
        _orig_reset = wrapper.reset
        def _reset_pinned():
            _orig_reset()
            wrapper.current_skill_id = pinned
        wrapper.reset = _reset_pinned
        _orig_act = wrapper.act
        def _act_pinned(obs):
            wrapper.current_skill_id = pinned  # pre-call, ensures the prompt uses `pinned`
            out = _orig_act(obs)
            wrapper.current_skill_id = pinned  # post-call, undoes any correction-rule flip
            return out
        wrapper.act = _act_pinned
        logger.info(f"pinned skill_id={pinned} (vote + correction-rule + reset "
                    f"overrides disabled — current_skill_id is a constant)")

    # Per-step action / prediction logger. Writes one JSONL file per (task,
    # instance, rollout) so the annotate_video.py post-processor can overlay
    # predictions on the sampled MP4. Log dir is created lazily on first write.
    # We hook wrapper.act to capture the outgoing action + wrapper state, and
    # wrapper.reset to close the previous log + open the next one.
    if args.action_log_dir:
        os.makedirs(args.action_log_dir, exist_ok=True)
        _log_state = {"fh": None, "step": 0}
        def _open_new_log():
            if _log_state["fh"]:
                _log_state["fh"].close()
            # File is opened lazily when we see the first obs with task_id
            _log_state["fh"] = None
            _log_state["step"] = 0
        _orig_reset2 = wrapper.reset
        def _reset_logged():
            _open_new_log()
            _orig_reset2()
        wrapper.reset = _reset_logged
        _orig_act2 = wrapper.act
        def _act_logged(obs):
            action = _orig_act2(obs)
            # Open file lazily on first act after reset, once we know the task/instance
            if _log_state["fh"] is None:
                task = str(obs.get("task_name", obs.get("task", "unknown_task")))
                # Best effort: derive an instance name from a "instance_id"-like key
                inst = obs.get("instance_id")
                if inst is None and "task_id" in obs:
                    inst = int(obs["task_id"][0]) if hasattr(obs["task_id"], "__iter__") else int(obs["task_id"])
                fname = f"{task}_{inst if inst is not None else 'unknown'}_step_log.jsonl"
                _log_state["fh"] = open(os.path.join(args.action_log_dir, fname), "a")
            # Serialise action tensor → list of floats
            act_np = action.detach().cpu().numpy() if hasattr(action, "detach") else np.asarray(action)
            record = {
                "step": _log_state["step"],
                "skill_id": int(getattr(wrapper, "current_skill_id", -1)),
                "subtask_progress": round(min(1.0, wrapper.step_count / max(1, wrapper.progress_horizon)), 4),
                "action": [round(float(x), 6) for x in act_np.tolist()[:23]],
            }
            import json as _j
            _log_state["fh"].write(_j.dumps(record) + "\n")
            _log_state["fh"].flush()
            _log_state["step"] += 1
            return action
        wrapper.act = _act_logged
        logger.info(f"per-step logging enabled → {args.action_log_dir}")

    logger.info(f"policy loaded. hostname={socket.gethostname()} port={args.port}")

    # replan_steps must match the wrapper's chunk stride so `need_obs=True` on
    # the client aligns with an actual re-infer on the server. If a caller
    # tunes execute_in_n_steps, they get a matching wire cadence for free.
    server = B1KChunkAwareServer(
        wrapper=wrapper,
        host=args.host,
        port=args.port,
        replan_steps=wrapper_config.execute_in_n_steps,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
