#!/usr/bin/env python3
"""Parse and compare the firmware's LOGD/LOGE logit dumps.

The bench firmware prints, for each dump:

    LOGD <label> <step> <n>
    <n> little-endian float32 values as 8 hex digits, 16 per line
    LOGE <label> <step>

This tool turns that into {label: [step][n]} float32 arrays, prints top-k
summaries, computes the sanity fingerprint used by the .m8 header, and can
compare two dumps bit-for-bit (the Step 2 loader check).

Usage:
    logit_dump.py summary <dump.log> [--top 5]
    logit_dump.py fingerprint <dump.log> <label> <step> [--top 32]
    logit_dump.py compare <a.log> <b.log> [--label int8] [--step 0]
"""
from __future__ import annotations

import argparse
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

# esp_rom_crc32_le(0, buf, len) on this target is the standard reflected
# CRC-32: init 0xFFFFFFFF and final xor 0xFFFFFFFF, because the ROM wrapper
# inverts at entry and exit. That is exactly zlib.crc32(). Verified on-device:
# the loader's header check rejected the non-inverting variant.
def crc32(data: bytes) -> int:
    return zlib.crc32(data) & 0xFFFFFFFF


def parse(path: Path) -> dict[str, list[np.ndarray]]:
    text = path.read_text(errors="replace").splitlines()
    out: dict[str, list[np.ndarray]] = {}
    i = 0
    while i < len(text):
        line = text[i].strip()
        if not line.startswith("LOGD "):
            i += 1
            continue
        _, label, step_s, n_s = line.split()
        step, n = int(step_s), int(n_s)
        i += 1
        words: list[str] = []
        while len(words) < n:
            if i >= len(text):
                raise ValueError(f"truncated dump at {label} {step}")
            words.extend(text[i].split())
            i += 1
        if len(words) != n:
            raise ValueError(f"bad word count at {label} {step}")
        vals = np.array([struct.unpack("<f", struct.pack("<I", int(w, 16)))[0]
                         for w in words], dtype=np.float32)
        out.setdefault(label, {})[step] = vals
        if i < len(text) and text[i].strip().startswith("LOGE "):
            i += 1
    return out


def parse_prompt(path: Path) -> list[int]:
    """Read the `PROMPT n=.. ids=a,b,c` line the bench firmware prints."""
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("PROMPT n=") and "ids=" in line:
            ids = line.split("ids=", 1)[1].strip()
            return [int(x) for x in ids.split(",") if x]
    return []


def parse_tokens(path: Path, label: str = "int8", rep: int = 0) -> list[int]:
    """Read the greedy continuation the firmware printed.

    Each line is `TOK <label> <rep> <index> <id>`; index restarts at 0 for every
    run. Only <label>/<rep> are collected, sorted by index.
    """
    out: dict[int, int] = {}
    for line in path.read_text(errors="replace").splitlines():
        parts = line.split()
        if len(parts) != 5 or parts[0] != "TOK" or parts[1] != label:
            continue
        try:
            r, idx, tid = int(parts[2]), int(parts[3]), int(parts[4])
        except ValueError:
            continue
        if r == rep:
            out[idx] = tid
    return [out[i] for i in sorted(out)]


def topk(v: np.ndarray, k: int) -> list[int]:
    # Descending value, ties broken by lower index (same as the firmware's
    # strict > comparison when scanning left to right).
    order = sorted(range(len(v)), key=lambda i: (-float(v[i]), i))
    return order[:k]


def fingerprint(v: np.ndarray, k: int) -> int:
    ids = topk(v, k)
    payload = b"".join(struct.pack("<f", float(v[i])) for i in ids)
    return crc32(payload)


def cmd_summary(args) -> int:
    d = parse(Path(args.dump))
    for label in sorted(d):
        steps = d[label]
        print(f"[{label}] steps={len(steps)} vocab={len(steps[min(steps)])}")
        for step in sorted(steps)[:args.steps]:
            v = steps[step]
            ids = topk(v, args.top)
            vals = ", ".join(f"{i}:{v[i]:.6g}" for i in ids)
            print(f"  step {step}: top{args.top} {vals}")
    return 0


def cmd_fingerprint(args) -> int:
    d = parse(Path(args.dump))
    v = d[args.label][args.step]
    fp = fingerprint(v, args.top)
    ids = topk(v, args.top)
    print(f"label={args.label} step={args.step} top={args.top}")
    print(f"ids={' '.join(str(i) for i in ids)}")
    print(f"crc32=0x{fp:08x}")
    print(f"top1={ids[0]} top1_logit={v[ids[0]]:.9g}")
    return 0


def cmd_compare(args) -> int:
    a = parse(Path(args.a))
    b = parse(Path(args.b))
    rc = 0
    labels = sorted(set(a) & set(b)) if args.label == "all" else [args.label]
    for label in labels:
        if label not in a or label not in b:
            print(f"{label}: missing in one dump")
            rc = 1
            continue
        sa, sb = a[label], b[label]
        common = sorted(set(sa) & set(sb))
        if args.step >= 0:
            common = [s for s in common if s == args.step]
        for step in common:
            va, vb = sa[step], sb[step]
            if va.shape != vb.shape:
                print(f"{label} {step}: shape {va.shape} vs {vb.shape}")
                rc = 1
                continue
            eq = int(np.sum(va == vb))
            if eq == len(va):
                print(f"{label} {step}: BIT-IDENTICAL ({eq}/{len(va)})")
            else:
                bad = np.flatnonzero(va != vb)
                d = np.abs(va[bad].astype(np.float64) - vb[bad].astype(np.float64))
                print(f"{label} {step}: {eq}/{len(va)} equal, "
                      f"max|d|={d.max():.6g}, first_bad={int(bad[0])}")
                rc = 1
    return rc


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("summary")
    s.add_argument("dump")
    s.add_argument("--top", type=int, default=5)
    s.add_argument("--steps", type=int, default=3)
    s.set_defaults(func=cmd_summary)

    f = sub.add_parser("fingerprint")
    f.add_argument("dump")
    f.add_argument("label")
    f.add_argument("step", type=int)
    f.add_argument("--top", type=int, default=32)
    f.set_defaults(func=cmd_fingerprint)

    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--label", default="int8")
    c.add_argument("--step", type=int, default=-1)
    c.set_defaults(func=cmd_compare)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
