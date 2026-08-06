"""Step rate from the log's `elapsed:` field, in fixed step windows.

tqdm's own `rate:` field is not usable here: it reflects the log-emission interval,
not the step interval, and overstated step time by ~2.5x during this run.
"""
import re, sys

def windows(job, w):
    pts = []
    try:
        f = open(f"/work6/markhsp/behavior-1k-solution/logs/slurm-{job}.out", errors="replace")
    except OSError:
        return [], None
    for line in f:
        m = re.search(r"Progress on: ([\d.]+)(k?)it/20\.0kit .*elapsed:([\d:]+)", line)
        if not m:
            continue
        v = float(m.group(1))
        step = int(v * 1000) if m.group(2) == "k" else int(v)
        q = [int(x) for x in m.group(3).split(":")]
        sec = q[0] * 3600 + q[1] * 60 + q[2] if len(q) == 3 else q[0] * 60 + q[1]
        pts.append((step, sec))
    out, prev = [], (pts[0] if pts else None)
    for step, sec in pts:
        if prev and step - prev[0] >= w:
            out.append((prev[0], step, (sec - prev[1]) / (step - prev[0])))
            prev = (step, sec)
    return out, (pts[-1] if pts else None)

if __name__ == "__main__":
    job, w = sys.argv[1], int(sys.argv[2])
    ws, last = windows(job, w)
    if not last:
        print("no-data"); sys.exit(0)
    tail = "  ".join(f"{a}->{b}:{r:.1f}s" for a, b, r in ws[-3:])
    print(f"step={last[0]} nwin={len(ws)} {tail}")
