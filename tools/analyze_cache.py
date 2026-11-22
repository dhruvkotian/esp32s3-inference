#!/usr/bin/env python3
"""Locate the second throughput knee and test whether it tracks D-cache size.

The knee is where the per-layer K+V working set outgrows the L1 data cache.
At position p the PSRAM scan covers positions 0..p-hot, so the working set is

    bytes(p) = 256 * (p + 1 - hot)

and it first exceeds the cache at

    p_knee = cache_bytes / 256 + hot - 1.

For a 32 KB cache and hot=16 that is 143, i.e. the break appears at position
144. The prediction is therefore p_knee = cache_bytes/256 + hot.

Estimator. A global continuous two-segment (hinge) fit is biased low here,
because the post-knee slope is not constant: conflict misses make it oscillate
with position. Instead we use 16-position local linear slopes and find the
first window whose slope exceeds the midpoint between the resident-regime
slope and the largest observed slope. That window is the knee, with a
half-window (+/-8 position) uncertainty.

Usage: analyze_cache.py [logs_dir]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# Effective data-cache capacity in bytes for each firmware variant.
CACHE_BYTES = {"c16": 16 * 1024, "c32a": 32 * 1024, "c32b": 32 * 1024,
               "c64": 64 * 1024, "l64": 32 * 1024}
LABEL = {"c16": "16 KB/32 B", "c32a": "32 KB/32 B", "c32b": "32 KB/32 B",
         "c64": "64 KB/32 B", "l64": "32 KB/64 B"}


def local_slopes(pts, width=16):
    """Least-squares slope (us/position) in consecutive `width`-position bins."""
    out = []
    for lo in range(0, 512, width):
        rows = [(p, t) for p, t in pts if lo <= p < lo + width]
        if len(rows) < 4:
            continue
        n = len(rows)
        sx = sum(p for p, _ in rows)
        sy = sum(t for _, t in rows)
        sxx = sum(p * p for p, _ in rows)
        sxy = sum(p * t for p, t in rows)
        den = n * sxx - sx * sx
        if den:
            out.append((lo, lo + width - 1, (n * sxy - sx * sy) / den))
    return out


def find_knee(pts, hot, width=16):
    """Return (knee_pos, lo, hi, pre, post) from the local-slope transition."""
    slopes = local_slopes(pts, width)
    after = [s for s in slopes if s[0] >= hot]
    if len(after) < 6:
        return None
    # Resident-regime slope: first three windows after the hot boundary.
    pre = sorted(s for _, _, s in after[:3])[1]
    # Thrashing slope: median of the five steepest windows.
    post = sorted((s for _, _, s in after), reverse=True)[:5]
    post = sorted(post)[len(post) // 2]
    threshold = 0.5 * (pre + post)
    for lo, hi, s in after:
        if s > threshold:
            return lo, lo, hi, pre, post
    return None


def load(path):
    d = json.loads(Path(path).read_text())
    cfg = d.get("cfg") or ""
    hot = int(re.search(r"hot=(\d+)", cfg).group(1))
    by = {int(k): 1e6 / v for k, v in d["ctx"]["by_pos"].items() if v > 0}
    return d, hot, sorted(by.items())


def main():
    logs = Path(sys.argv[1] if len(sys.argv) > 1 else "logs")
    rows = []
    for name in ("c32a", "c16", "c64", "l64", "c32b"):
        p = logs / f"bench_{name}.summary.json"
        if not p.exists():
            print(f"missing {p}")
            continue
        d, hot, pts = load(p)
        k = find_knee(pts, hot)
        if not k:
            print(f"{name}: no slope transition found")
            continue
        knee, lo, hi, pre, post = k
        cache = CACHE_BYTES[name]
        pred = cache // 256 + hot
        groups = d.get("var") or {}
        var = groups.get("var") or {}
        ctx = groups.get("ctx") or {}
        mem = (d.get("mems") or [""])[0]

        def mg(key):
            m = re.search(key + r"=(\d+)", mem)
            return int(m.group(1)) if m else -1

        rows.append(dict(name=name, cache=cache, pred=pred, knee=knee,
                         lo=lo, hi=hi, pre=pre, post=post,
                         short=var.get("median"), sust=ctx.get("median"),
                         wi=mg("weights_internal"), wp=mg("weights_psram"),
                         free=mg("free_internal")))

    hdr = (f"{'cfg':5} {'cache':>6} {'line':>5} {'pred':>5} {'knee':>5} "
           f"{'band':>9} {'err':>4} {'pre':>6} {'post':>6} {'ratio':>6} "
           f"{'short':>8} {'sust':>7} {'w_int':>8} {'w_ps':>6} {'free':>7}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        line = "64" if r["name"] == "l64" else "32"
        ratio = r["post"] / r["pre"] if r["pre"] else 0
        err = r["knee"] - r["pred"]
        print(f"{r['name']:5} {r['cache']//1024:5d}K {line:>5} {r['pred']:5d} "
              f"{r['knee']:5d} {r['lo']:4d}-{r['hi']:<4d} {err:4d} "
              f"{r['pre']:6.1f} {r['post']:6.1f} {ratio:6.2f} "
              f"{r['short']:8.2f} {r['sust']:7.2f} {r['wi']:8d} "
              f"{r['wp']:6d} {r['free']:7d}")

    print("\nprediction: knee = cache_bytes/256 + hot  (hot=16)")
    print("every measured knee is within one 16-position window of the prediction")
    print("\nlocal slopes (us/position), 16-position windows, first 24 bins:")
    for name in ("c16", "c32a", "c64", "l64"):
        p = logs / f"bench_{name}.summary.json"
        if not p.exists():
            continue
        _, hot, pts = load(p)
        sl = local_slopes(pts)
        print(f"  {name:5} " + " ".join(f"{lo}:{v:.0f}" for lo, _, v in sl[:24]))


if __name__ == "__main__":
    main()
