#!/usr/bin/env python3
"""Pack a legacy llama2.c v0 fp32 checkpoint into a prequantized .m8 blob.

The quantizer reproduces firmware/main/llm.c:qmat_fill() exactly:

    mx   = max(|row|)
    s    = mx / 127.0f          (1.0f if mx == 0)
    inv  = 1.0f / s
    q[i] = clamp(trunc((row[i] * inv) + copysign(0.5, row[i] * inv)), -127, 127)
    pad the row to a 16-byte stride with zeros

All arithmetic is done in float32 so the bytes match the C implementation.
`tools/quant_ref.c` contains the same routine compiled for the host; pass its
output with --ref to prove byte equality.

Usage:
    pack_model_m8.py --model model/stories260K.bin --tokenizer model/tok512.bin \
        --out model/stories260k-q8.m8 --name "TinyStories 260K q8" \
        [--sanity-dump logs/golden-logits-2026-11-25.log] [--ref build/quant_ref.bin]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import struct
import sys
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MAGIC = b"M8B1"
VERSION = 1
HEADER_BYTES = 4096
TENSOR_ENTRY_BYTES = 36
MAX_TENSORS = 16
MAX_BLOB_BYTES = 512 * 1024
ROW_ALIGN = 16

F_Q8 = 1
F_F32 = 2
F_REQUIRED = 4
NO_SCALE = 0xFFFFFFFF

# Tensor ids. The eight matrices are row-stacked across layers exactly as
# llm_quantize() does today, so they bind straight into the existing qmat_t.
TID_TOK_EMB = 1
TID_WQ = 2
TID_WK = 3
TID_WV = 4
TID_WO = 5
TID_W1 = 6
TID_W2 = 7
TID_W3 = 8
TID_RMS_ATT = 9
TID_RMS_FFN = 10
TID_RMS_FINAL = 11

# Fixed layout inside the 4 KiB header.
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
OFF_HEADS = 62
OFF_KV_HEADS = 64
OFF_VOCAB = 66
OFF_SEQ = 68
OFF_CTX = 70
OFF_QUANT_BITS = 72
OFF_ROW_ALIGN = 73
OFF_TENSOR_COUNT = 74
OFF_Q_BYTES = 76
OFF_SCALE_BYTES = 80
OFF_NORM_BYTES = 84
OFF_TOK_OFF = 88
OFF_TOK_BYTES = 92
OFF_SANITY_LEN = 96
OFF_SANITY_TOPK = 97
OFF_SANITY_CRC = 100
OFF_SANITY_PROMPT = 104   # u16[8]
OFF_SANITY_IDS = 120      # u16[8]
OFF_NAME = 140            # char[32]
OFF_BUILD_ID = 172        # char[32]
OFF_TENSORS = 256


@dataclass
class Tensor:
    tid: int
    name: str
    kind: str                 # "q8" or "f32"
    rows: int
    cols: int
    stride: int               # q stride in bytes, or row stride in bytes for f32
    q: np.ndarray | None = None
    scale: np.ndarray | None = None
    data: np.ndarray | None = None
    data_offset: int = 0
    scale_offset: int = NO_SCALE


@dataclass
class Checkpoint:
    dim: int
    hidden: int
    layers: int
    heads: int
    kv_heads: int
    vocab: int
    seq: int
    kv_dim: int
    head_size: int
    tensors: dict[str, np.ndarray] = field(default_factory=dict)
    file_sha256: str = ""


def crc32(data: bytes) -> int:
    # esp_rom_crc32_le(0, buf, len) is the standard reflected CRC-32 (the ROM
    # wrapper inverts at entry and exit), i.e. exactly zlib.crc32().
    return zlib.crc32(data) & 0xFFFFFFFF


def align(n: int, a: int) -> int:
    return (n + a - 1) // a * a


def load_checkpoint(path: Path) -> Checkpoint:
    raw = path.read_bytes()
    if len(raw) < 28:
        raise ValueError("checkpoint shorter than the 28-byte header")
    dim, hidden, nl, nh, nkvh, vocab, seq = struct.unpack_from("<7i", raw, 0)
    if vocab <= 0:
        raise ValueError("unshared classifier (negative vocab) is not supported")
    if dim % nh or nh % nkvh:
        raise ValueError("invalid head geometry")
    kv_dim = dim * nkvh // nh
    head_size = dim // nh

    layout = [
        ("tok_embeddings", vocab * dim),
        ("rms_att", nl * dim),
        ("wq", nl * dim * dim),
        ("wk", nl * kv_dim * dim),
        ("wv", nl * kv_dim * dim),
        ("wo", nl * dim * dim),
        ("rms_ffn", nl * dim),
        ("w1", nl * hidden * dim),
        ("w2", nl * hidden * dim),
        ("w3", nl * hidden * dim),
        ("rms_final", dim),
        ("freqs_cos", seq * head_size // 2),
        ("freqs_sin", seq * head_size // 2),
    ]
    off = 28
    cp = Checkpoint(dim=dim, hidden=hidden, layers=nl, heads=nh,
                    kv_heads=nkvh, vocab=vocab, seq=seq, kv_dim=kv_dim,
                    head_size=head_size)
    for name, n in layout:
        end = off + 4 * n
        if end > len(raw):
            raise ValueError(f"checkpoint truncated in {name}")
        if not name.startswith("freqs_"):
            cp.tensors[name] = np.frombuffer(raw, dtype="<f4", count=n,
                                             offset=off).astype(np.float32)
        off = end
    if off != len(raw):
        raise ValueError(f"layout mismatch: consumed {off} of {len(raw)} bytes")
    cp.file_sha256 = hashlib.sha256(raw).hexdigest()
    return cp


def quantize_rows(w: np.ndarray, rows: int, cols: int) -> tuple[np.ndarray, np.ndarray]:
    """Bit-compatible with llm.c:qmat_fill(). Returns (q int8 [rows,stride], scales f32)."""
    stride = align(cols, ROW_ALIGN)
    src = w.reshape(rows, cols)
    q = np.zeros((rows, stride), dtype=np.int8)
    scales = np.zeros(rows, dtype=np.float32)
    half = np.float32(0.5)
    maxv = np.float32(127.0)
    one = np.float32(1.0)

    # float32 max over |row|
    mx = np.abs(src).max(axis=1).astype(np.float32)
    s = np.where(mx > np.float32(0.0), mx / maxv, one).astype(np.float32)
    inv = (one / s).astype(np.float32)
    t = (src * inv[:, None]).astype(np.float32)
    # trunc(x + copysign(0.5, x)) is round-half-away-from-zero
    r = np.where(t >= np.float32(0.0), t + half, t - half).astype(np.float32)
    qi = r.astype(np.int32)                 # truncates toward zero
    qi = np.clip(qi, -127, 127)
    q[:, :cols] = qi.astype(np.int8)
    scales[:] = s
    return q, scales


def build_tensors(cp: Checkpoint) -> list[Tensor]:
    nl, dim, hid, kv = cp.layers, cp.dim, cp.hidden, cp.kv_dim
    specs = [
        (TID_TOK_EMB, "tok_embeddings", "q8", cp.vocab, dim),
        (TID_WQ, "wq", "q8", nl * dim, dim),
        (TID_WK, "wk", "q8", nl * kv, dim),
        (TID_WV, "wv", "q8", nl * kv, dim),
        (TID_WO, "wo", "q8", nl * dim, dim),
        (TID_W1, "w1", "q8", nl * hid, dim),
        (TID_W2, "w2", "q8", nl * dim, hid),
        (TID_W3, "w3", "q8", nl * hid, dim),
        (TID_RMS_ATT, "rms_att", "f32", nl, dim),
        (TID_RMS_FFN, "rms_ffn", "f32", nl, dim),
        (TID_RMS_FINAL, "rms_final", "f32", 1, dim),
    ]
    out: list[Tensor] = []
    for tid, name, kind, rows, cols in specs:
        w = cp.tensors[name]
        if kind == "q8":
            q, s = quantize_rows(w, rows, cols)
            out.append(Tensor(tid=tid, name=name, kind=kind, rows=rows,
                              cols=cols, stride=align(cols, ROW_ALIGN),
                              q=q, scale=s))
        else:
            if w.size != rows * cols:
                raise ValueError(f"{name}: expected {rows * cols}, got {w.size}")
            out.append(Tensor(tid=tid, name=name, kind=kind, rows=rows,
                              cols=cols, stride=cols * 4,
                              data=w.reshape(rows, cols).astype("<f4")))
    return out


def load_logit_dump():
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("logit_dump", here / "logit_dump.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_payload(tensors: list[Tensor], tokenizer: bytes) -> tuple[bytearray, int, int, int, int]:
    payload = bytearray()
    q_bytes = 0
    scale_bytes = 0
    norm_bytes = 0

    for t in tensors:
        if t.kind != "q8":
            continue
        off = align(len(payload), ROW_ALIGN)
        payload.extend(b"\0" * (off - len(payload)))
        t.data_offset = off
        payload.extend(t.q.tobytes(order="C"))
        q_bytes += t.rows * t.stride

    for t in tensors:
        if t.kind != "q8":
            continue
        off = align(len(payload), 4)
        payload.extend(b"\0" * (off - len(payload)))
        t.scale_offset = off
        payload.extend(t.scale.astype("<f4").tobytes())
        scale_bytes += t.rows * 4

    for t in tensors:
        if t.kind != "f32":
            continue
        off = align(len(payload), 4)
        payload.extend(b"\0" * (off - len(payload)))
        t.data_offset = off
        payload.extend(t.data.astype("<f4").tobytes(order="C"))
        norm_bytes += t.rows * t.cols * 4

    tok_off = align(len(payload), 4)
    payload.extend(b"\0" * (tok_off - len(payload)))
    payload.extend(tokenizer)

    total = align(len(payload), HEADER_BYTES)
    payload.extend(b"\0" * (total - len(payload)))
    return payload, q_bytes, scale_bytes, norm_bytes, tok_off


def encode_name(s: str, n: int) -> bytes:
    b = s.encode("utf-8")[:n - 1]
    return b + b"\0" * (n - len(b))


def pack(cp: Checkpoint, tokenizer: bytes, name: str, build_id: str,
         sanity: dict | None) -> bytes:
    tensors = build_tensors(cp)
    payload, q_bytes, scale_bytes, norm_bytes, tok_off = build_payload(tensors, tokenizer)
    if len(payload) > MAX_BLOB_BYTES - HEADER_BYTES:
        raise ValueError(f"payload {len(payload)} B exceeds M8_MAX_BYTES")

    hdr = bytearray(HEADER_BYTES)
    hdr[OFF_MAGIC:OFF_MAGIC + 4] = MAGIC
    struct.pack_into("<H", hdr, OFF_VERSION, VERSION)
    struct.pack_into("<H", hdr, OFF_HEADER_BYTES, HEADER_BYTES)
    struct.pack_into("<I", hdr, OFF_FILE_BYTES, HEADER_BYTES + len(payload))
    struct.pack_into("<I", hdr, OFF_PAYLOAD_BYTES, len(payload))
    struct.pack_into("<I", hdr, OFF_FLAGS, 1)  # tokenizer embedded
    hdr[OFF_PAYLOAD_SHA:OFF_PAYLOAD_SHA + 32] = hashlib.sha256(payload).digest()
    struct.pack_into("<H", hdr, OFF_DIM, cp.dim)
    struct.pack_into("<H", hdr, OFF_HIDDEN, cp.hidden)
    struct.pack_into("<H", hdr, OFF_LAYERS, cp.layers)
    struct.pack_into("<H", hdr, OFF_HEADS, cp.heads)
    struct.pack_into("<H", hdr, OFF_KV_HEADS, cp.kv_heads)
    struct.pack_into("<H", hdr, OFF_VOCAB, cp.vocab)
    struct.pack_into("<H", hdr, OFF_SEQ, cp.seq)
    struct.pack_into("<H", hdr, OFF_CTX, cp.seq)
    struct.pack_into("<B", hdr, OFF_QUANT_BITS, 8)
    struct.pack_into("<B", hdr, OFF_ROW_ALIGN, ROW_ALIGN)
    struct.pack_into("<H", hdr, OFF_TENSOR_COUNT, len(tensors))
    struct.pack_into("<I", hdr, OFF_Q_BYTES, q_bytes)
    struct.pack_into("<I", hdr, OFF_SCALE_BYTES, scale_bytes)
    struct.pack_into("<I", hdr, OFF_NORM_BYTES, norm_bytes)
    struct.pack_into("<I", hdr, OFF_TOK_OFF, tok_off)
    struct.pack_into("<I", hdr, OFF_TOK_BYTES, len(tokenizer))
    hdr[OFF_NAME:OFF_NAME + 32] = encode_name(name, 32)
    hdr[OFF_BUILD_ID:OFF_BUILD_ID + 32] = encode_name(build_id, 32)

    if sanity:
        ids = sanity["prompt_ids"]
        top = sanity["ids"]
        struct.pack_into("<B", hdr, OFF_SANITY_LEN, len(ids))
        struct.pack_into("<B", hdr, OFF_SANITY_TOPK, sanity["topk"])
        struct.pack_into("<I", hdr, OFF_SANITY_CRC, sanity["crc32"])
        for i, v in enumerate(ids[:8]):
            struct.pack_into("<H", hdr, OFF_SANITY_PROMPT + 2 * i, v)
        for i, v in enumerate(top[:8]):
            struct.pack_into("<H", hdr, OFF_SANITY_IDS + 2 * i, v)

    for i, t in enumerate(tensors):
        base = OFF_TENSORS + i * TENSOR_ENTRY_BYTES
        flags = (F_Q8 if t.kind == "q8" else F_F32) | F_REQUIRED
        struct.pack_into("<HHIIIIII", hdr, base,
                         t.tid, flags, t.rows, t.cols, t.stride,
                         t.data_offset,
                         t.rows * t.stride if t.kind == "q8" else t.rows * t.cols * 4,
                         t.scale_offset)
        struct.pack_into("<I", hdr, base + 28,
                         t.rows * 4 if t.kind == "q8" else 0)
        struct.pack_into("<B", hdr, base + 32, 2)  # tier: any (advisory)

    struct.pack_into("<I", hdr, OFF_HEADER_CRC, 0)
    struct.pack_into("<I", hdr, OFF_HEADER_CRC, crc32(bytes(hdr)))
    return bytes(hdr) + bytes(payload)


def parse_header(blob: bytes) -> dict:
    if len(blob) < HEADER_BYTES or blob[:4] != MAGIC:
        raise ValueError("bad magic")
    version, hbytes = struct.unpack_from("<HH", blob, OFF_VERSION)
    if version != VERSION or hbytes != HEADER_BYTES:
        raise ValueError(f"unsupported version/header {version}/{hbytes}")
    file_bytes, payload_bytes = struct.unpack_from("<II", blob, OFF_FILE_BYTES)
    if file_bytes != len(blob) or file_bytes != HEADER_BYTES + payload_bytes:
        raise ValueError(f"size mismatch file={file_bytes} len={len(blob)} payload={payload_bytes}")
    stored = struct.unpack_from("<I", blob, OFF_HEADER_CRC)[0]
    tmp = bytearray(blob[:HEADER_BYTES])
    struct.pack_into("<I", tmp, OFF_HEADER_CRC, 0)
    if crc32(bytes(tmp)) != stored:
        raise ValueError("header CRC mismatch")
    payload = blob[HEADER_BYTES:]
    if hashlib.sha256(payload).digest() != blob[OFF_PAYLOAD_SHA:OFF_PAYLOAD_SHA + 32]:
        raise ValueError("payload SHA-256 mismatch")
    count = struct.unpack_from("<H", blob, OFF_TENSOR_COUNT)[0]
    if count > MAX_TENSORS:
        raise ValueError("too many tensors")
    tensors = []
    for i in range(count):
        base = OFF_TENSORS + i * TENSOR_ENTRY_BYTES
        tid, flags, rows, cols, stride, doff, dbytes, soff = \
            struct.unpack_from("<HHIIIIII", blob, base)
        sbytes = struct.unpack_from("<I", blob, base + 28)[0]
        tier = blob[base + 32]
        if flags & F_Q8:
            if stride % ROW_ALIGN or stride < cols:
                raise ValueError(f"tensor {tid}: bad stride {stride}")
            if dbytes != rows * stride:
                raise ValueError(f"tensor {tid}: dbytes {dbytes} != {rows * stride}")
            if soff != NO_SCALE and sbytes != rows * 4:
                raise ValueError(f"tensor {tid}: bad scale bytes")
            if doff % ROW_ALIGN:
                raise ValueError(f"tensor {tid}: data offset not 16-aligned")
            if soff != NO_SCALE and soff % 4:
                raise ValueError(f"tensor {tid}: scale offset not 4-aligned")
        else:
            if dbytes != rows * cols * 4:
                raise ValueError(f"tensor {tid}: bad f32 size")
            if doff % 4:
                raise ValueError(f"tensor {tid}: data offset not 4-aligned")
        if doff + dbytes > payload_bytes:
            raise ValueError(f"tensor {tid}: data out of range")
        if soff != NO_SCALE and soff + sbytes > payload_bytes:
            raise ValueError(f"tensor {tid}: scales out of range")
        tensors.append(dict(tid=tid, flags=flags, rows=rows, cols=cols,
                            stride=stride, data_offset=doff,
                            data_bytes=dbytes, scale_offset=soff,
                            scale_bytes=sbytes, tier=tier))
    tok_off = struct.unpack_from("<I", blob, OFF_TOK_OFF)[0]
    tok_bytes = struct.unpack_from("<I", blob, OFF_TOK_BYTES)[0]
    if tok_off % 4 or tok_off + tok_bytes > payload_bytes:
        raise ValueError("tokenizer range invalid")
    return dict(version=version, file_bytes=file_bytes,
                payload_bytes=payload_bytes, tensors=tensors,
                tok_off=tok_off, tok_bytes=tok_bytes)


def ref_bytes(cp: Checkpoint) -> bytes:
    """Byte layout expected from tools/quant_ref.c."""
    out = bytearray(b"Q8REF1\0\0")
    specs = [
        (cp.vocab, cp.dim), (cp.layers * cp.dim, cp.dim),
        (cp.layers * cp.kv_dim, cp.dim), (cp.layers * cp.kv_dim, cp.dim),
        (cp.layers * cp.dim, cp.dim), (cp.layers * cp.hidden, cp.dim),
        (cp.layers * cp.dim, cp.hidden), (cp.layers * cp.hidden, cp.dim),
    ]
    names = ["tok_embeddings", "wq", "wk", "wv", "wo", "w1", "w2", "w3"]
    out.extend(struct.pack("<I", len(specs)))
    for (rows, cols), name in zip(specs, names):
        q, s = quantize_rows(cp.tensors[name], rows, cols)
        out.extend(struct.pack("<III", rows, cols, q.shape[1]))
        out.extend(q.tobytes())
        out.extend(s.astype("<f4").tobytes())
    return bytes(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="model")
    ap.add_argument("--build-id", default="")
    ap.add_argument("--sanity-dump", default="")
    ap.add_argument("--sanity-label", default="int8")
    ap.add_argument("--sanity-step", type=int, default=0)
    ap.add_argument("--sanity-topk", type=int, default=32)
    ap.add_argument("--ref", default="")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    cp = load_checkpoint(Path(args.model))
    tok = Path(args.tokenizer).read_bytes()
    build_id = args.build_id or f"packer1 {cp.file_sha256[:12]}"

    sanity = None
    if args.sanity_dump:
        ld = load_logit_dump()
        d = ld.parse(Path(args.sanity_dump))
        v = d[args.sanity_label][args.sanity_step]
        prompt_ids = ld.parse_prompt(Path(args.sanity_dump))
        if not prompt_ids:
            raise SystemExit("sanity dump has no PROMPT line; cannot bind the fingerprint")
        # The header stores the greedy continuation, not the top-k of the
        # prompt's last logits: the device replays prompt + ids and checks each
        # argmax, which is exactly what the firmware printed as TOK lines.
        greedy = ld.parse_tokens(Path(args.sanity_dump), args.sanity_label, args.sanity_step)
        if len(greedy) < 8:
            raise SystemExit(
                f"sanity dump has {len(greedy)} TOK {args.sanity_label} "
                f"{args.sanity_step} lines; the header needs at least 8")
        sanity = dict(
            prompt_ids=prompt_ids,
            ids=greedy,
            topk=args.sanity_topk,
            crc32=ld.fingerprint(v, args.sanity_topk),
        )

    blob = pack(cp, tok, args.name, build_id, sanity)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(blob)

    hdr = parse_header(blob)
    print(f"wrote {out} ({len(blob)} B, {len(blob) / 1024:.1f} KiB)")
    print(f"  sha256 {hashlib.sha256(blob).hexdigest()}")
    print(f"  tensors {len(hdr['tensors'])}, payload {hdr['payload_bytes']} B, "
          f"tokenizer {hdr['tok_bytes']} B at {hdr['tok_off']}")
    for t in hdr["tensors"]:
        print(f"  id={t['tid']:2d} {'q8' if t['flags'] & F_Q8 else 'f32':3s} "
              f"{t['rows']}x{t['cols']} stride={t['stride']} "
              f"data@{t['data_offset']}+{t['data_bytes']} "
              f"scale@{t['scale_offset'] if t['scale_offset'] != NO_SCALE else '-'}")
    if sanity:
        print(f"  sanity prompt={sanity['prompt_ids']} top{args.sanity_topk} "
              f"crc=0x{sanity['crc32']:08x} ids={sanity['ids']}")

    if args.ref:
        got = ref_bytes(cp)
        want = Path(args.ref).read_bytes()
        if got != want:
            n = min(len(got), len(want))
            bad = next((i for i in range(n) if got[i] != want[i]), n)
            print(f"REF MISMATCH: ours {len(got)} B, ref {len(want)} B, "
                  f"first diff at {bad}", file=sys.stderr)
            return 1
        print(f"  quantizer matches tools/quant_ref.c byte-for-byte ({len(got)} B)")

    if args.check:
        blob2 = pack(cp, tok, args.name, build_id, sanity)
        if blob2 != blob:
            print("DETERMINISM FAILURE: two packs differ", file=sys.stderr)
            return 1
        print("  deterministic: second pack is byte-identical")

    return 0


if __name__ == "__main__":
    sys.exit(main())
