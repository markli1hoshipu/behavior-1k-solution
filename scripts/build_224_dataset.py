"""Rebuild the B1K 2026 v3.0 dataset with 224x224 videos.

Only the videos change. Frame counts, episode boundaries, timestamps, parquet rows and
annotations are all reused as-is, so `data/`, `meta/` and `annotations/` are hardlinked
(free, same filesystem) and only `meta/info.json` is rewritten to declare the new video
shapes.

Why: the model consumes 224x224, so the source 720x720 / 480x480 frames are decoded at
~10x / ~4.6x the necessary pixel count and then resized per sample on the CPU. Measured
on this machine, decode+resize costs 86.5 ms/sample against ~49 ms for raw decode alone
-- the resize alone nearly doubles it. Baking the resize in removes both costs for every
epoch of every future run.

Correctness requirement: the rebuilt video must preserve frame count and presentation
timestamps exactly. `meta/episodes` stores per-camera from_timestamp/to_timestamp per
episode and the parquet rows are matched to frames by timestamp, so a single dropped or
shifted frame silently misaligns observations from actions. --verify checks this per file.

Usage:
  # size the job / validate on one file
  python scripts/build_224_dataset.py --limit 1 --verify
  # full run
  python scripts/build_224_dataset.py --workers 200
"""
import argparse
import json
import os
import pathlib
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import av
import numpy as np

SRC = pathlib.Path("/work6/markhsp/datasets/behavior-1k-2026-challenge-demos")
DST = pathlib.Path("/work6/markhsp/datasets/2026-challenge-demos-224")
SIZE = 224
GOP = 8          # match the source's keyframe spacing: cheap random access
CRF = 24         # PSNR 36.3 dB (zed, min 34.0) at 0.47x source size; see logs/build_224.log
CODEC = "libx264"


def transcode(rel_path: str, verify: bool = False) -> dict:
    """Re-encode one video file to SIZE x SIZE, preserving frame count and PTS."""
    src = SRC / rel_path
    dst = DST / rel_path
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".mp4.tmp")

    t0 = time.perf_counter()
    n_in = 0
    with av.open(str(src)) as ic:
        istream = ic.streams.video[0]
        istream.thread_type = "AUTO"

        # format is explicit: the ".mp4.tmp" suffix defeats extension sniffing.
        with av.open(str(tmp), "w", format="mp4") as oc:
            ostream = oc.add_stream(CODEC, rate=istream.average_rate)
            ostream.width = SIZE
            ostream.height = SIZE
            ostream.pix_fmt = "yuv420p"
            # Preserve the source timebase so copied PTS keep their meaning.
            ostream.time_base = istream.time_base
            ostream.codec_context.time_base = istream.time_base
            ostream.codec_context.gop_size = GOP
            ostream.codec_context.options = {
                "crf": str(CRF),
                "preset": "veryfast",
                # Deterministic, seek-friendly: no open GOP, no scene-cut reordering of
                # keyframes, so keyframes land exactly every GOP frames.
                "g": str(GOP),
                "keyint_min": str(GOP),
                "sc_threshold": "0",
                "x264-params": f"keyint={GOP}:min-keyint={GOP}:scenecut=0:open-gop=0",
            }

            for frame in ic.decode(istream):
                out = frame.reformat(width=SIZE, height=SIZE, format="yuv420p")
                out.pts = frame.pts
                out.time_base = istream.time_base
                for pkt in ostream.encode(out):
                    oc.mux(pkt)
                n_in += 1
            for pkt in ostream.encode():
                oc.mux(pkt)

    dt = time.perf_counter() - t0
    result = {
        "path": rel_path,
        "frames_in": n_in,
        "seconds": dt,
        "src_bytes": src.stat().st_size,
        "dst_bytes": tmp.stat().st_size,
    }

    if verify:
        # Frame count and PTS sequence must match exactly.
        def pts_list(p):
            out = []
            with av.open(str(p)) as c:
                s = c.streams.video[0]
                for f in c.decode(s):
                    out.append(f.pts)
            return out

        src_pts, dst_pts = pts_list(src), pts_list(tmp)
        result["frames_out"] = len(dst_pts)
        result["pts_match"] = src_pts == dst_pts
        if not result["pts_match"]:
            result["pts_first_mismatch"] = next(
                (i for i, (a, b) in enumerate(zip(src_pts, dst_pts)) if a != b), None
            )
        # Keyframe spacing in the output.
        with av.open(str(tmp)) as c:
            s = c.streams.video[0]
            kf = [i for i, p in enumerate(c.demux(s)) if p.pts is not None and p.is_keyframe]
        gaps = [b - a for a, b in zip(kf, kf[1:])]
        result["gop_mode"] = max(set(gaps), key=gaps.count) if gaps else None
        # Quality: PSNR of output vs the source downscaled the same way.
        with av.open(str(src)) as c1, av.open(str(tmp)) as c2:
            s1, s2 = c1.streams.video[0], c2.streams.video[0]
            d1, d2 = c1.decode(s1), c2.decode(s2)
            psnrs = []
            for i, (f1, f2) in enumerate(zip(d1, d2)):
                a = f1.reformat(width=SIZE, height=SIZE, format="rgb24").to_ndarray().astype(np.float64)
                b = f2.to_ndarray(format="rgb24").astype(np.float64)
                mse = ((a - b) ** 2).mean()
                psnrs.append(float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse))
                if i >= 200:
                    break
            result["psnr_mean"] = float(np.mean(psnrs))
            result["psnr_min"] = float(np.min(psnrs))

    os.replace(tmp, dst)
    return result


def link_tree(name: str) -> int:
    """Hardlink SRC/name into DST/name (contents are unchanged by the rebuild)."""
    n = 0
    for src_path in (SRC / name).rglob("*"):
        if not src_path.is_file():
            continue
        rel = src_path.relative_to(SRC)
        dst_path = DST / rel
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if dst_path.exists():
            continue
        try:
            os.link(src_path, dst_path)
        except OSError:
            shutil.copy2(src_path, dst_path)
        n += 1
    return n


def write_info_json() -> None:
    """Write DST/meta/info.json: RGB reshaped to SIZE, depth features dropped.

    Two things this must get right:

    1. UNLINK BEFORE WRITING. link_tree() has already hardlinked every meta file,
       including info.json, so opening the destination "w" truncates the *shared inode*
       and rewrites the pristine official file in SRC. That happened once and had to be
       restored from the Hub.

    2. Drop the depth_linear features. We never downloaded depth (2 TB, unused: the model
       is RGB + proprio). LeRobotDataset._check_cached_episodes_sufficient() walks
       meta.video_keys and calls video_path.exists() for every requested episode, so a
       declared-but-absent depth video makes it declare the local copy insufficient and
       fall back to downloading from the Hub on every single dataset construction.
    """
    with open(SRC / "meta/info.json") as f:
        info = json.load(f)

    features = info.get("features", {})
    changed, dropped = [], []
    for key in list(features):
        feat = features[key]
        if feat.get("dtype") != "video":
            continue
        if key.startswith("observation.depth"):
            del features[key]
            dropped.append(key)
            continue
        if key.startswith("observation.rgb"):
            old = feat.get("shape")
            feat["shape"] = [SIZE, SIZE, old[2] if old and len(old) > 2 else 3]
            vi = feat.get("info") or {}
            if "video.height" in vi:
                vi["video.height"] = SIZE
            if "video.width" in vi:
                vi["video.width"] = SIZE
            if "video.codec" in vi:
                vi["video.codec"] = "h264"
            feat["info"] = vi
            changed.append(f"{key}: {old} -> {feat['shape']}")

    dst = DST / "meta/info.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.unlink(missing_ok=True)  # break the hardlink; never write through to SRC
    with open(dst, "w") as f:
        json.dump(info, f, indent=4)
    for c in changed:
        print(f"  info.json  {c}")
    for d in dropped:
        print(f"  info.json  dropped {d} (never downloaded)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=200)
    ap.add_argument("--limit", type=int, default=None, help="only N video files (sizing/validation)")
    ap.add_argument("--verify", action="store_true", help="check frame count / PTS / GOP / PSNR")
    ap.add_argument("--skip-link", action="store_true", help="do not (re)link data/meta/annotations")
    ap.add_argument("--redo", action="store_true",
                    help="re-transcode files that already exist in DST (default: skip them)")
    args = ap.parse_args()

    videos = sorted(
        str(p.relative_to(SRC)) for p in (SRC / "videos").rglob("*.mp4")
    )
    total_bytes = sum((SRC / v).stat().st_size for v in videos)

    # Resume by default. A full pass is ~44 h over 5286 files; without this, any kill
    # restarts from zero. Outputs land via .tmp + os.replace, so a present .mp4 is
    # complete by construction — there are no half-written files to distrust.
    # Use --redo after changing SIZE/CRF/GOP, since those leave valid-but-stale outputs.
    if not args.redo:
        before = len(videos)
        videos = [v for v in videos if not (DST / v).exists()]
        skipped = before - len(videos)
        if skipped:
            print(f"断点续跑: 跳过 {skipped} 个已完成文件, 剩余 {len(videos)} 个"
                  f"  (--redo 可强制重做)\n")
    print(f"源: {SRC}")
    print(f"目标: {DST}")
    print(f"视频文件 {len(videos)} 个, {total_bytes/2**40:.2f} TiB")
    print(f"参数: {SIZE}x{SIZE}  {CODEC}  crf={CRF}  gop={GOP}\n")

    if args.limit:
        videos = videos[: args.limit]
        print(f"[limit] 只处理 {len(videos)} 个文件\n")

    if not args.skip_link:
        print("硬链接 data / meta / annotations ...")
        for name in ("data", "meta", "annotations"):
            if (SRC / name).exists():
                print(f"  {name}: {link_tree(name)} 个文件")
        for extra in ("README.md", "LICENSE"):
            if (SRC / extra).is_file() and not (DST / extra).exists():
                os.link(SRC / extra, DST / extra)
        write_info_json()
        print()

    done = frames = src_b = dst_b = 0
    t0 = time.perf_counter()
    failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(transcode, v, args.verify): v for v in videos}
        for fut in as_completed(futs):
            v = futs[fut]
            try:
                r = fut.result()
            except Exception as e:  # noqa: BLE001
                failures.append((v, repr(e)))
                print(f"  FAIL {v}: {e}", flush=True)
                continue
            done += 1
            frames += r["frames_in"]
            src_b += r["src_bytes"]
            dst_b += r["dst_bytes"]
            if args.verify:
                print(f"  {r['path']}")
                print(f"    帧数 in={r['frames_in']} out={r.get('frames_out')}  "
                      f"PTS 完全一致={r.get('pts_match')}  输出 GOP={r.get('gop_mode')}")
                print(f"    PSNR mean={r.get('psnr_mean'):.2f} dB  min={r.get('psnr_min'):.2f} dB")
                print(f"    体积 {r['src_bytes']/2**20:.1f} → {r['dst_bytes']/2**20:.1f} MiB "
                      f"({r['dst_bytes']/r['src_bytes']:.3f}x)")
                print(f"    耗时 {r['seconds']:.1f}s  = {r['frames_in']/r['seconds']:.0f} 帧/秒")
            elif done % 50 == 0 or done == len(videos):
                el = time.perf_counter() - t0
                rate = done / el
                eta = (len(videos) - done) / rate if rate else 0
                # NB: frames/seconds count only *completed* files. Every worker holds a
                # ~237k-frame file, so until the pipeline has cycled once these figures
                # under-report badly -- the first reading showed 1033 frames/s against a
                # true ~5000, and an ETA 5x too pessimistic. Treat early lines as a floor.
                warm = "  [流水线未填满，以下为下限]" if done < args.workers else ""
                print(f"  {done}/{len(videos)} 文件  {frames/1e6:.1f}M 帧  "
                      f"{frames/el:.0f} 帧/秒  体积 {dst_b/max(1,src_b):.3f}x  "
                      f"ETA {eta/3600:.1f} h{warm}", flush=True)

    el = time.perf_counter() - t0
    print(f"\n完成 {done}/{len(videos)}  失败 {len(failures)}")
    print(f"总帧数 {frames/1e6:.1f}M   耗时 {el/3600:.2f} h   平均 {frames/max(el,1e-9):.0f} 帧/秒")
    if src_b:
        print(f"体积 {src_b/2**30:.1f} → {dst_b/2**30:.1f} GiB  ({dst_b/src_b:.3f}x)")
    if args.limit and done:
        # Project the full job from this sample.
        all_bytes = total_bytes
        frac = src_b / all_bytes if all_bytes else 0
        if frac:
            print(f"\n按此样本外推全量: 约 {el/frac/3600:.1f} 核·小时的单文件耗时基准, "
                  f"预计输出 {dst_b/frac/2**30:.0f} GiB")
    for v, e in failures[:10]:
        print(f"  FAILED {v}: {e}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
