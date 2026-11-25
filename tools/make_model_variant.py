#!/usr/bin/env python3
"""Create a deterministic synthetic variant of a llama2.c v0 checkpoint.

This is a test fixture, not a trained model. It scales every weight matrix and
RMS norm by a fixed factor and copies the RoPE tables unchanged, so the result
has the same shapes and tokenizer but different logits. Use it to exercise the
runtime model switch (Step 3) with two distinct parameter sets.

The device's sanity fingerprint is bound to the int8 quantized weights, so a
variant must first be packed without a fingerprint, booted once with
DUMP_LOGITS=1, and re-packed with that dump. See docs/MODEL-LIBRARY-DESIGN.md.

Usage:
    make_model_variant.py --model model/stories260K.bin \
        --out /tmp/stories260K_alt.bin --scale 1.03
"""
from __future__ import annotations

import argparse
import hashlib
import struct
from pathlib import Path

import numpy as np


def layout_for(dim: int, hidden: int, layers: int, heads: int,
               kv_heads: int, vocab: int, seq: int) -> list[tuple[str, int]]:
    kv_dim = dim * kv_heads // heads
    head_size = dim // heads
    return [
        ("tok_embeddings", vocab * dim),
        ("rms_att", layers * dim),
        ("wq", layers * dim * dim),
        ("wk", layers * kv_dim * dim),
        ("wv", layers * kv_dim * dim),
        ("wo", layers * dim * dim),
        ("rms_ffn", layers * dim),
        ("w1", layers * hidden * dim),
        ("w2", layers * hidden * dim),
        ("w3", layers * hidden * dim),
        ("rms_final", dim),
        ("freqs_cos", seq * head_size // 2),
        ("freqs_sin", seq * head_size // 2),
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="source llama2.c checkpoint")
    ap.add_argument("--out", required=True, help="variant checkpoint")
    ap.add_argument("--scale", type=float, default=1.03)
    ap.add_argument("--bias", type=float, default=0.0)
    args = ap.parse_args()

    src = Path(args.model)
    raw = bytearray(src.read_bytes())
    if len(raw) < 28:
        raise SystemExit("checkpoint shorter than the 28-byte header")
    dim, hidden, layers, heads, kv_heads, vocab, seq = struct.unpack_from(
        "<7i", raw, 0)
    if min(dim, hidden, layers, heads, kv_heads, vocab, seq) <= 0:
        raise SystemExit("invalid checkpoint geometry")
    if dim % heads or heads % kv_heads:
        raise SystemExit("invalid head geometry")

    scale = np.float32(args.scale)
    bias = np.float32(args.bias)
    off = 28
    changed = 0
    for name, count in layout_for(dim, hidden, layers, heads, kv_heads,
                                  vocab, seq):
        end = off + 4 * count
        if end > len(raw):
            raise SystemExit(f"checkpoint truncated in {name}")
        if not name.startswith("freqs_"):
            w = np.frombuffer(raw, dtype="<f4", count=count, offset=off)
            w = (w.astype(np.float32) * scale + bias).astype("<f4")
            raw[off:end] = w.tobytes()
            changed += 1
        off = end
    if off != len(raw):
        raise SystemExit(f"layout mismatch: consumed {off} of {len(raw)} bytes")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(raw)
    print(f"wrote {out} ({len(raw)} B), {changed} tensors scaled by "
          f"{args.scale} bias {args.bias}")
    print(f"sha256 {hashlib.sha256(raw).hexdigest()}")
    print("# synthetic variant for switch tests; not a trained model")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
