#!/usr/bin/env python3
"""Build a synthetic MoE sidecar from packed .m8 model blobs.

The sidecar holds FFN-only experts. Expert 0 is the FFN of the first `--blob`,
expert 1 the FFN of the second, and so on. The backbone (embedding, attention,
norms) is not duplicated: it stays in the active .m8 model. Router weights are
deterministic synthetic floats, so routing quality is explicitly out of scope;
this fixture exists to validate the copy path and measure its cost.

Format (little-endian, 4 KiB header) is documented in firmware/main/moe.h.

Usage:
    make_moe_sidecar.py \
        --blob /tmp/opencode/m8/stories260k-q8.m8 \
        --blob /tmp/opencode/step3/alt.m8 \
        --out /tmp/opencode/moe/moe.bin --seed 2026 --check
"""
from __future__ import annotations

import argparse
import hashlib
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

M8_HEADER = 4096
M8_ENTRY = 36
M8_MAX_TENSORS = 16
M8_NO_SCALE = 0xFFFFFFFF

MOE_MAGIC = b"MOE1"
MOE_VERSION = 1
MOE_HEADER = 4096
FLAG_ROUTER_F32 = 1
FLAG_SYNTHETIC = 2

T_FFN = (6, 7, 8)  # w1, w2, w3

# Header offsets, mirrored from firmware/main/moe.h.
OFF_MAGIC = 0
OFF_VERSION = 4
OFF_HEADER_BYTES = 6
OFF_FILE_BYTES = 8
OFF_PAYLOAD_BYTES = 12
OFF_FLAGS = 16
OFF_HEADER_CRC = 20
OFF_PAYLOAD_SHA = 24
OFF_DIM = 56
OFF_HIDDEN = 58
OFF_LAYERS = 60
OFF_EXPERTS = 62
OFF_ROW_ALIGN = 64
OFF_MODEL_CRC = 68
OFF_ROUTER_OFF = 72
OFF_ROUTER_BYTES = 76
OFF_EXPERT_BYTES = 80
OFF_EXPERT_OFF = 84
OFF_W1Q = 88
OFF_W1S = 96
OFF_W2Q = 104
OFF_W2S = 112
OFF_W3Q = 120
OFF_W3S = 128
OFF_NAME = 136
OFF_BUILD = 168


def crc32(data: bytes) -> int:
    # esp_rom_crc32_le(0, ...) == zlib.crc32(); see docs/MODEL-LIBRARY-DESIGN.md.
    return zlib.crc32(data) & 0xFFFFFFFF


def align(n: int, a: int) -> int:
    return (n + a - 1) & ~(a - 1)


def parse_m8(path: Path) -> dict:
    blob = path.read_bytes()
    if len(blob) < M8_HEADER or blob[0:4] != b"M8B1":
        raise ValueError(f"{path}: bad magic or short file")
    version, hbytes = struct.unpack_from("<HH", blob, 4)
    if version != 1 or hbytes != M8_HEADER:
        raise ValueError(f"{path}: version/header {version}/{hbytes}")
    file_bytes, payload_bytes = struct.unpack_from("<II", blob, 8)
    if file_bytes != len(blob) or file_bytes != M8_HEADER + payload_bytes:
        raise ValueError(f"{path}: size mismatch")
    flags = struct.unpack_from("<I", blob, 16)[0]
    if not (flags & 1):
        raise ValueError(f"{path}: tokenizer not embedded")
    hdr = bytearray(blob[:M8_HEADER])
    stored_crc = struct.unpack_from("<I", hdr, 20)[0]
    struct.pack_into("<I", hdr, 20, 0)
    if crc32(bytes(hdr)) != stored_crc:
        raise ValueError(f"{path}: header CRC32 mismatch")
    payload = blob[M8_HEADER:]
    if hashlib.sha256(payload).digest() != blob[24:56]:
        raise ValueError(f"{path}: payload SHA-256 mismatch")

    dim, hidden, layers, heads, kv_heads, vocab, seq, ctx = \
        struct.unpack_from("<8H", blob, 56)
    count = struct.unpack_from("<H", blob, 74)[0]
    if not 0 < count <= M8_MAX_TENSORS:
        raise ValueError(f"{path}: tensor_count {count}")

    tensors: dict[int, dict] = {}
    for i in range(count):
        base = 256 + i * M8_ENTRY
        tid, tflags, rows, cols, stride, doff, dbytes, soff = \
            struct.unpack_from("<HHIIIIII", blob, base)
        sbytes = struct.unpack_from("<I", blob, base + 28)[0]
        if tid in tensors:
            raise ValueError(f"{path}: duplicate tensor id {tid}")
        if tid <= 8 and not (tflags & 1):
            raise ValueError(f"{path}: tensor {tid} is not q8")
        if doff + dbytes > payload_bytes:
            raise ValueError(f"{path}: tensor {tid} outside the payload")
        if (tflags & 1) and soff + sbytes > payload_bytes:
            raise ValueError(f"{path}: tensor {tid} scales outside the payload")
        tensors[tid] = dict(rows=rows, cols=cols, stride=stride,
                            q=payload[doff:doff + dbytes],
                            s=payload[soff:soff + sbytes])

    for tid in T_FFN:
        if tid not in tensors:
            raise ValueError(f"{path}: missing FFN tensor {tid}")
    stride1 = align(dim, 16)
    if tensors[6]["rows"] != layers * hidden or tensors[6]["cols"] != dim or \
            tensors[6]["stride"] != stride1:
        raise ValueError(f"{path}: w1 geometry does not match")
    if tensors[8]["rows"] != layers * hidden or tensors[8]["cols"] != dim or \
            tensors[8]["stride"] != stride1:
        raise ValueError(f"{path}: w3 geometry does not match")
    if tensors[7]["rows"] != layers * dim or tensors[7]["cols"] != hidden or \
            tensors[7]["stride"] != align(hidden, 16):
        raise ValueError(f"{path}: w2 geometry does not match")

    return dict(path=path, blob=blob, dim=dim, hidden=hidden, layers=layers,
                heads=heads, kv_heads=kv_heads, vocab=vocab, seq=seq, ctx=ctx,
                tensors=tensors,
                name=blob[140:172].split(b"\0")[0].decode("ascii", "replace"),
                build=blob[172:204].split(b"\0")[0].decode("ascii", "replace"),
                sanity_crc=struct.unpack_from("<I", blob, 100)[0])


def build_expert_block(m8: dict) -> tuple[bytes, dict]:
    block = bytearray()
    offs: dict[str, int] = {}

    def place(key: str, data: bytes, alignment: int) -> None:
        while len(block) % alignment:
            block.append(0)
        offs[key + "_off"] = len(block)
        offs[key + "_bytes"] = len(data)
        block.extend(data)

    t = m8["tensors"]
    place("w1q", t[6]["q"], 16)
    place("w2q", t[7]["q"], 16)
    place("w3q", t[8]["q"], 16)
    place("w1s", t[6]["s"], 4)
    place("w2s", t[7]["s"], 4)
    place("w3s", t[8]["s"], 4)
    return bytes(block), offs


def build_sidecar(blobs: list[Path], name: str, build: str, seed: int,
                  router_scale: float) -> tuple[bytes, dict]:
    models = [parse_m8(p) for p in blobs]
    dim = models[0]["dim"]
    hidden = models[0]["hidden"]
    layers = models[0]["layers"]
    for m in models:
        if (m["dim"], m["hidden"], m["layers"]) != (dim, hidden, layers):
            raise ValueError(f"{m['path']}: geometry differs from expert 0")
    experts = len(models)
    if not 0 < experts <= 4:
        raise ValueError("1 to 4 experts are supported")

    blocks = []
    offs = None
    for m in models:
        block, block_offs = build_expert_block(m)
        if offs is None:
            offs = block_offs
        elif block_offs != offs or len(block) != len(blocks[0]):
            raise ValueError(f"{m['path']}: expert layout differs")
        blocks.append(block)
    expert_bytes = len(blocks[0])
    assert offs is not None

    rng = np.random.default_rng(seed)
    router = rng.normal(0.0, router_scale,
                        size=(layers, experts, dim)).astype("<f4")
    router_bytes = router.tobytes()

    payload = bytearray(router_bytes)
    while len(payload) % 16:
        payload.append(0)
    router_off = 0
    expert_off = len(payload)
    for block in blocks:
        payload.extend(block)

    hdr = bytearray(MOE_HEADER)
    hdr[OFF_MAGIC:OFF_MAGIC + 4] = MOE_MAGIC
    struct.pack_into("<HH", hdr, OFF_VERSION, MOE_VERSION, MOE_HEADER)
    struct.pack_into("<II", hdr, OFF_FILE_BYTES, MOE_HEADER + len(payload),
                     len(payload))
    struct.pack_into("<I", hdr, OFF_FLAGS, FLAG_ROUTER_F32 | FLAG_SYNTHETIC)
    struct.pack_into("<8H", hdr, OFF_DIM, dim, hidden, layers, experts,
                     0, 0, 0, 0)
    hdr[OFF_ROW_ALIGN] = 16
    struct.pack_into("<I", hdr, OFF_MODEL_CRC, models[0]["sanity_crc"])
    struct.pack_into("<I", hdr, OFF_ROUTER_OFF, router_off)
    struct.pack_into("<I", hdr, OFF_ROUTER_BYTES, len(router_bytes))
    struct.pack_into("<I", hdr, OFF_EXPERT_BYTES, expert_bytes)
    struct.pack_into("<I", hdr, OFF_EXPERT_OFF, expert_off)
    for base, key in ((OFF_W1Q, "w1q"), (OFF_W1S, "w1s"),
                      (OFF_W2Q, "w2q"), (OFF_W2S, "w2s"),
                      (OFF_W3Q, "w3q"), (OFF_W3S, "w3s")):
        struct.pack_into("<II", hdr, base, offs[key + "_off"],
                         offs[key + "_bytes"])
    hdr[OFF_NAME:OFF_NAME + 32] = name.encode("ascii")[:32].ljust(32, b"\0")
    hdr[OFF_BUILD:OFF_BUILD + 32] = build.encode("ascii")[:32].ljust(32, b"\0")
    # SHA and header CRC are computed last.
    hdr[OFF_PAYLOAD_SHA:OFF_PAYLOAD_SHA + 32] = hashlib.sha256(payload).digest()
    struct.pack_into("<I", hdr, OFF_HEADER_CRC, 0)
    struct.pack_into("<I", hdr, OFF_HEADER_CRC, crc32(bytes(hdr)))

    meta = dict(dim=dim, hidden=hidden, layers=layers, experts=experts,
                router_bytes=len(router_bytes), expert_off=expert_off,
                expert_bytes=expert_bytes, offsets=offs, models=models,
                router=router)
    return bytes(hdr) + bytes(payload), meta


def verify(blob: bytes, meta: dict) -> None:
    if len(blob) < MOE_HEADER or blob[0:4] != MOE_MAGIC:
        raise ValueError("built sidecar has bad magic")
    file_bytes, payload_bytes = struct.unpack_from("<II", blob, OFF_FILE_BYTES)
    if file_bytes != len(blob) or payload_bytes != len(blob) - MOE_HEADER:
        raise ValueError("built sidecar has bad sizes")
    hdr = bytearray(blob[:MOE_HEADER])
    stored = struct.unpack_from("<I", hdr, OFF_HEADER_CRC)[0]
    struct.pack_into("<I", hdr, OFF_HEADER_CRC, 0)
    if crc32(bytes(hdr)) != stored:
        raise ValueError("built sidecar header CRC mismatch")
    payload = blob[MOE_HEADER:]
    if hashlib.sha256(payload).digest() != blob[24:56]:
        raise ValueError("built sidecar payload SHA mismatch")

    # Every expert block must carry the exact FFN bytes of its source blob.
    t_off = meta["expert_off"]
    t_bytes = meta["expert_bytes"]
    for e, model in enumerate(meta["models"]):
        block = payload[t_off + e * t_bytes:t_off + (e + 1) * t_bytes]
        t = model["tensors"]
        for key, tid in (("w1q", 6), ("w2q", 7), ("w3q", 8)):
            off = meta["offsets"][key + "_off"]
            n = meta["offsets"][key + "_bytes"]
            if block[off:off + n] != t[tid]["q"]:
                raise ValueError(
                    f"expert {e} {key} does not match its source blob")
        for key, tid in (("w1s", 6), ("w2s", 7), ("w3s", 8)):
            off = meta["offsets"][key + "_off"]
            n = meta["offsets"][key + "_bytes"]
            if block[off:off + n] != t[tid]["s"]:
                raise ValueError(
                    f"expert {e} {key} does not match its source blob")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blob", action="append", required=True,
                    help="packed .m8 source; repeat in expert order")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="synthetic-moe")
    ap.add_argument("--build", default="moe-tool 1")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--router-scale", type=float, default=0.05)
    ap.add_argument("--check", action="store_true",
                    help="rebuild and confirm byte-identical output")
    args = ap.parse_args()

    paths = [Path(p) for p in args.blob]
    blob, meta = build_sidecar(paths, args.name, args.build, args.seed,
                               args.router_scale)
    verify(blob, meta)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(blob)

    print(f"wrote {out} ({len(blob)} B, {len(blob) / 1024:.1f} KiB)")
    print(f"  sha256 {hashlib.sha256(blob).hexdigest()}")
    print(f"  geometry dim={meta['dim']} hidden={meta['hidden']} "
          f"layers={meta['layers']} experts={meta['experts']}")
    print(f"  router {meta['router_bytes']} B @ 0, "
          f"expert blocks {meta['expert_bytes']} B @ {meta['expert_off']}")
    for key in ("w1q", "w2q", "w3q", "w1s", "w2s", "w3s"):
        print(f"  {key}: off={meta['offsets'][key + '_off']} "
              f"bytes={meta['offsets'][key + '_bytes']}")
    for i, m in enumerate(meta["models"]):
        print(f"  expert {i}: {m['path'].name} name=\"{m['name']}\" "
              f"sanity=0x{m['sanity_crc']:08x}")

    if args.check:
        blob2, _ = build_sidecar(paths, args.name, args.build, args.seed,
                                 args.router_scale)
        if blob2 != blob:
            print("DETERMINISM FAILURE: two builds differ", file=sys.stderr)
            return 1
        print("  deterministic: second build is byte-identical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
