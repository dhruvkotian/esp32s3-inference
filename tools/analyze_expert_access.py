#!/usr/bin/env python3
"""Summarise the PSRAM-resident expert access benchmark.

Reads a UART capture containing the `EA_*` lines printed by
firmware/main/expert_access_bench.c and prints a per-configuration table,
separating cold runs from warm repeats.

Usage:
    python3 tools/analyze_expert_access.py logs/expert_access.log
    python3 tools/analyze_expert_access.py logs/expert_access.log --csv out.csv

The benchmark runs three memory-access modes over identical weights:

    A  per-layer staging    PSRAM -> 79,296 B SRAM slice -> compute
    B  direct PSRAM         compute straight from the mapped block
    C  tiled streaming      PSRAM -> 1/2/4/8/16 KB tile -> compute

All modes move the same number of bytes out of PSRAM per expert forward, so the
MB/s column is comparable across them.

Run 0 of each configuration is cold (the D-cache is evicted first); runs 1 and
later are warm repeats. They are reported separately because pooling them hides
the difference. Compare modes on total_us: for A that is copy plus compute, for
B it is the direct compute alone.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def parse(lines: list[str]) -> tuple[list[dict], list[str]]:
    """Split the capture into EA_RUN records and the remaining EA_* lines."""
    runs: list[dict] = []
    meta: list[str] = []
    for raw in lines:
        line = raw.strip()
        if not line.startswith("EA_"):
            continue
        if line.startswith("EA_RUN "):
            rec: dict[str, object] = {}
            for token in line[len("EA_RUN ") :].split():
                if "=" not in token:
                    continue
                key, value = token.split("=", 1)
                rec[key] = value
            if "mode" in rec:
                runs.append(rec)
        else:
            meta.append(line)
    return runs, meta


def as_int(rec: dict, key: str, default: int = 0) -> int:
    try:
        return int(rec.get(key, default))
    except (TypeError, ValueError):
        return default


def cache_state(rec: dict) -> str:
    """Prefer the explicit cold flag; fall back to run 0 for older captures.

    The firmware emits cold=1 when the D-cache was evicted before that run.
    Without the flag (EA_COLD_EVERY=0 captures still set it, but older logs may
    not) run 0 is the cold measurement and every later run is a warm repeat.
    """
    if "cold" in rec:
        return "cold" if as_int(rec, "cold") else "warm"
    return "cold" if as_int(rec, "run") == 0 else "warm"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log", nargs="?", help="capture file; stdin when omitted")
    ap.add_argument("--csv", type=Path, help="write the grouped table as CSV")
    ap.add_argument(
        "--min-runs",
        type=int,
        default=2,
        help="flag warm groups with fewer repeats than this (default: 2)",
    )
    args = ap.parse_args(argv)

    if args.log:
        text = Path(args.log).read_text(encoding="utf-8", errors="replace")
    else:
        text = sys.stdin.read()

    runs, meta = parse(text.splitlines())
    if not runs:
        print("no EA_RUN lines found", file=sys.stderr)
        return 1

    for line in meta:
        print(line)
    print()

    groups: dict[tuple[str, int, str, str], list[dict]] = defaultdict(list)
    for rec in runs:
        key = (
            str(rec.get("mode")),
            as_int(rec, "tile"),
            str(rec.get("pattern")),
            cache_state(rec),
        )
        groups[key].append(rec)

    header = (
        f"{'mode':<11} {'tile':>6} {'pattern':<12} {'state':<5} {'runs':>4} "
        f"{'copy_us':>8} {'comp_us':>8} {'total_us':>9} {'MB/s':>7} {'bytes':>9}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    thin: list[tuple[str, int, str, str, int]] = []
    order = {"cold": 0, "warm": 1}
    for key in sorted(groups, key=lambda k: (k[0], k[1], k[2], order.get(k[3], 9))):
        mode, tile, pattern, state = key
        recs = groups[key]
        copy_us = int(statistics.median(as_int(r, "copy_us") for r in recs))
        comp_us = int(statistics.median(as_int(r, "comp_us") for r in recs))
        total_us = int(statistics.median(as_int(r, "total_us") for r in recs))
        nbytes = int(statistics.median(as_int(r, "psram_bytes") for r in recs))
        mbps = (nbytes / total_us) if total_us else 0.0
        print(
            f"{mode:<11} {tile:>6} {pattern:<12} {state:<5} {len(recs):>4} "
            f"{copy_us:>8} {comp_us:>8} {total_us:>9} {mbps:>7.1f} {nbytes:>9}"
        )
        rows.append(
            {
                "mode": mode,
                "tile": tile,
                "pattern": pattern,
                "state": state,
                "runs": len(recs),
                "copy_us": copy_us,
                "comp_us": comp_us,
                "total_us": total_us,
                "mbps": round(mbps, 2),
                "psram_bytes": nbytes,
            }
        )
        # A cold group is a single run by construction; only warm repeats can
        # be under-sampled.
        if state == "warm" and len(recs) < args.min_runs:
            thin.append((mode, tile, pattern, state, len(recs)))

    print()
    print("Reading it:")
    print("  - Compare A with B on total_us at the same pattern and state. They")
    print("    move the same PSRAM bytes, so any gap is the SRAM round trip")
    print("    against cached 128-bit loads from the mapping.")
    print("  - Cold and warm rows answer different questions. The cold row is")
    print("    the first touch of the expert; the warm row is what a steady-state")
    print("    token loop sees.")
    print("  - Mode C stages weight tiles but reads the row scales directly from")
    print("    PSRAM, so A vs C is not a pure copy-versus-stream comparison.")
    print("  - C-prefetch rows include the software pre-touch time in copy_us.")
    print("  - 'alternating' switches the expert between layers of one forward")
    print("    pass; it does not yet model per-token router instability.")

    if thin:
        print()
        print(f"Under-sampled warm groups (fewer than {args.min_runs} repeats):")
        for mode, tile, pattern, state, n in thin:
            print(f"  {mode} tile={tile} {pattern} {state}: {n}")

    if args.csv:
        with args.csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
