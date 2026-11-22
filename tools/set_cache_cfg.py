#!/usr/bin/env python3
"""Set the ESP32-S3 data-cache size and line size in sdkconfig.

Usage: set_cache_cfg.py <size_kb:16|32|64> <line_bytes:16|32|64>

The ESP32-S3 Kconfig represents the size as a choice (16/32/64 KB) plus a
numeric CONFIG_ESP32S3_DATA_CACHE_SIZE. The 16 KB choice intentionally keeps
SIZE=0x8000 and occupies half the cache at startup, returning 48 KB to the
heap; see components/esp_system/port/cpu_start.c. This helper edits both the
choice and the numeric symbol so the build is self-consistent.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

SIZES = {
    16: ("CONFIG_ESP32S3_DATA_CACHE_16KB", "0x8000"),
    32: ("CONFIG_ESP32S3_DATA_CACHE_32KB", "0x8000"),
    64: ("CONFIG_ESP32S3_DATA_CACHE_64KB", "0x10000"),
}
LINES = {
    16: ("CONFIG_ESP32S3_DATA_CACHE_LINE_16B", "16"),
    32: ("CONFIG_ESP32S3_DATA_CACHE_LINE_32B", "32"),
    64: ("CONFIG_ESP32S3_DATA_CACHE_LINE_64B", "64"),
}


def set_symbol(text: str, sym: str, value: str | None) -> str:
    """Set sym=value, or comment it out when value is None."""
    pat = re.compile(r"^(?:# )?" + re.escape(sym) + r"=.*$", re.M)
    if value is None:
        line = f"# {sym} is not set"
    else:
        line = f"{sym}={value}"
    if pat.search(text):
        return pat.sub(line, text, count=1)
    if re.search(r"^# " + re.escape(sym) + r" is not set$", text, re.M):
        if value is None:
            return text
    if not text.endswith("\n"):
        text += "\n"
    return text + line + "\n"


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    size_kb, line_b = int(sys.argv[1]), int(sys.argv[2])
    if size_kb not in SIZES or line_b not in LINES:
        print("unsupported size/line", file=sys.stderr)
        return 2

    path = Path("sdkconfig")
    text = path.read_text()

    for kb, (sym, _) in SIZES.items():
        text = set_symbol(text, sym, "y" if kb == size_kb else None)
    size_sym, size_hex = SIZES[size_kb]
    text = set_symbol(text, "CONFIG_ESP32S3_DATA_CACHE_SIZE", size_hex)

    for b, (sym, _) in LINES.items():
        text = set_symbol(text, sym, "y" if b == line_b else None)
    line_sym, line_val = LINES[line_b]
    text = set_symbol(text, "CONFIG_ESP32S3_DATA_CACHE_LINE_SIZE", line_val)

    # Associativity is held at 8 ways for every variant.
    text = set_symbol(text, "CONFIG_ESP32S3_DATA_CACHE_4WAYS", None)
    text = set_symbol(text, "CONFIG_ESP32S3_DATA_CACHE_8WAYS", "y")
    text = set_symbol(text, "CONFIG_ESP32S3_DCACHE_ASSOCIATED_WAYS", "8")

    path.write_text(text)
    print(f"sdkconfig: data cache {size_kb} KB ({size_hex}), line {line_b} B, 8-way")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
