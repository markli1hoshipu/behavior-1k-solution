"""Fetch JackLiu0406/b1k-2026-dataset-224 into datasets/2026-challenge-demos-224-hf.

Phased with retries: anonymous access to ~6k small files gets HTTP 429, and a failure in
one phase must not block the others. snapshot_download resumes by etag, so retries are cheap.
"""
import os, time, sys
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
from huggingface_hub import snapshot_download

RID = "JackLiu0406/b1k-2026-dataset-224"
LOCAL = "/work6/markhsp/datasets/2026-challenge-demos-224-hf"
PHASES = [("meta", ["meta/**", ".gitattributes"], 4),
          ("data", ["data/**"], 6),
          ("videos", ["videos/**"], 6)]
for name, pats, w in PHASES:
    for att in range(1, 9):
        try:
            print(f"[{name}] attempt {att}", flush=True)
            snapshot_download(repo_id=RID, repo_type="dataset", local_dir=LOCAL,
                              allow_patterns=pats, max_workers=w, tqdm_class=None)
            print(f"[{name}] OK", flush=True); break
        except Exception as e:
            print(f"[{name}] {type(e).__name__}: {str(e)[:100]}", flush=True)
            if att == 8: print(f"[{name}] GIVING UP", flush=True); sys.exit(1)
            time.sleep(min(60, 5*att))
print("DONE", flush=True)
