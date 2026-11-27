#!/usr/bin/env python3
"""Summarise the PSRAM-resident expert access benchmark.

Reads a UART capture containing the `EA_*` lines printed by
firmware/main/expert_access_bench.c and prints a per-configuration table.

Usage:
    python3 tools/analyze_expert_access.py logs/expert_access.log
    idf.py monitor | tee logs/expert_access.log
    python3 tools/analyze_expert_access.py logs/expert_access.log --csv out.csv

The benchmark runs three memory-access modes over identical weights:

    A  per-layer staging    PSRAM -> 79,296 B SRAM slice -> compute
    B  direct PSRAM         compute straight from the mapped block
    C  tiled streaming      PSRAM -> 1/2/4/8/16 KB tile -> compute

All modes move the same number of bytes out of PSRAM per expert forward, so the
MB/s column is comparable across them.
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


def as_float(rec: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(rec.get(key, default))
    except (TypeError, ValueError):
        return default


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log", nargs="?", help="capture file; stdin when omitted")
    ap.add_argument("--csv", type=Path, help="write the grouped table as CSV")
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

    groups: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
    for rec in runs:
        key = (str(rec.get("mode")), as_int(rec, "tile"), str(rec.get("pattern")))
        groups[key].append(rec)

    header = (
        f"{'mode':<11} {'tile':>6} {'pattern':<12} {'runs':>4} "
        f"{'copy_us':>8} {'comp_us':>8} {'total_us':>9} {'MB/s':>7} {'bytes':>9}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for key in sorted(groups, key=lambda k: (k[0], k[1], k[2])):
        mode, tile, pattern = key
        recs = groups[key]
        copy_us = statistics.median(as_int(r, "copy_us") for r in recs)
        comp_us = statistics.median(as_int(r, "comp_us") for r in recs)
        total_us = statistics.median(as_int(r, "total_us") for r in recs)
        nbytes = statistics.median(as_int(r, "psram_bytes") for r in recs)
        mbps = (nbytes / total_us) if total_us else 0.0
        print(
            f"{mode:<11} {tile:>6} {pattern:<12} {len(recs):>4} "
            f"{copy_us:>8} {comp_us:>8} {total_us:>9} {mbps:>7.1f} {nbytes:>9}"
        )
        rows.append(
            {
                "mode": mode,
                "tile": tile,
                "pattern": pattern,
                "runs": len(recs),
                "copy_us": copy_us,
                "comp_us": comp_us,
                "total_us": total_us,
                "mbps": round(mbps, 2),
                "psram_bytes": nbytes,
            }
        )

    print()
    print("Reading it:")
    print("  - Compare mode A with mode B at the same pattern. A and B move the")
    print("    same PSRAM bytes, so any gap is the cost or benefit of the SRAM")
    print("    round trip versus cached 128-bit loads from the mapping.")
    print("  - Compare mode C tiles against A. C only wins if bounded tiles keep")
    print("    the pipeline busier than one large per-layer copy.")
    print("  - The C-prefetch rows include the software-touch time in copy_us.")
    print("  - run=0 is cold (D-cache evicted); run=1.. are warm repeats.")

    if args.csv:
        with args.csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
