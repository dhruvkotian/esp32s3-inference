#!/usr/bin/env python3
"""Wrap a file for the 'model' / 'tok' flash partitions.

Layout: uint32 little-endian length, then the raw file bytes.
Used only as a fallback when the SD card filesystem cannot be mounted.
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: make_part_image.py <input> <output>")
        return 2
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    data = src.read_bytes()
    dst.write_bytes(struct.pack("<I", len(data)) + data)
    print(f"{dst}: {len(data)} B payload + 4 B header = {dst.stat().st_size} B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
