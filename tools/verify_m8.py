#!/usr/bin/env python3
"""Independently verify an .m8 blob.

This deliberately re-implements the header parser instead of importing the
packer, so a bug in one side is not hidden by the other.

Checks:
  1. magic/version/size/CRC32/SHA-256
  2. tensor table bounds, alignment, row sizes
  3. dequantise every q8 tensor and compare against the source checkpoint
     (max error must be within half a quantisation step)
  4. rebuild a legacy llama2.c v0 fp32 file from the blob and run the pure
     Python reference forward, then compare greedy ids against a golden
     LOGD dump from the device

Usage:
    verify_m8.py --blob build/stories260k-q8.m8 \
        --tokenizer model/tok512.bin \
        --golden logs/golden-logits-2026-11-25.log
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import struct
import sys
import zlib
import tempfile
from pathlib import Path

import numpy as np

MAGIC = b"M8B1"
VERSION = 1
HEADER_BYTES = 4096
ENTRY = 36
MAX_TENSORS = 16
NO_SCALE = 0xFFFFFFFF
F_Q8 = 1
F_F32 = 2

TID = {1: "tok_embeddings", 2: "wq", 3: "wk", 4: "wv", 5: "wo",
       6: "w1", 7: "w2", 8: "w3", 9: "rms_att", 10: "rms_ffn",
       11: "rms_final"}


def crc32(data: bytes) -> int:
    # esp_rom_crc32_le(0, buf, len) is the standard reflected CRC-32 (the ROM
    # wrapper inverts at entry and exit), i.e. exactly zlib.crc32().
    return zlib.crc32(data) & 0xFFFFFFFF


def parse_blob(blob: bytes) -> dict:
    if len(blob) < HEADER_BYTES:
        raise ValueError("blob shorter than the header")
    if blob[0:4] != MAGIC:
        raise ValueError(f"bad magic {blob[0:4]!r}")
    version, hbytes = struct.unpack_from("<HH", blob, 4)
    if version != VERSION:
        raise ValueError(f"version {version}")
    if hbytes != HEADER_BYTES:
        raise ValueError(f"header_bytes {hbytes}")
    file_bytes, payload_bytes = struct.unpack_from("<II", blob, 8)
    if file_bytes != len(blob):
        raise ValueError(f"file_bytes {file_bytes} != actual {len(blob)}")
    if file_bytes != HEADER_BYTES + payload_bytes:
        raise ValueError("file/payload size mismatch")
    flags = struct.unpack_from("<I", blob, 16)[0]
    stored_crc = struct.unpack_from("<I", blob, 20)[0]
    tmp = bytearray(blob[:HEADER_BYTES])
    struct.pack_into("<I", tmp, 20, 0)
    if crc32(bytes(tmp)) != stored_crc:
        raise ValueError("header CRC32 mismatch")
    payload = blob[HEADER_BYTES:]
    if hashlib.sha256(payload).digest() != blob[24:56]:
        raise ValueError("payload SHA-256 mismatch")

    dim, hidden, nl, nh, nkvh, vocab, seq, ctx = struct.unpack_from("<8H", blob, 56)
    quant_bits, row_align = struct.unpack_from("<BB", blob, 72)
    count = struct.unpack_from("<H", blob, 74)[0]
    q_bytes, s_bytes, n_bytes = struct.unpack_from("<III", blob, 76)
    tok_off, tok_bytes = struct.unpack_from("<II", blob, 88)
    sanity_len, sanity_topk = struct.unpack_from("<BB", blob, 96)
    sanity_crc = struct.unpack_from("<I", blob, 100)[0]
    sanity_prompt = struct.unpack_from("<8H", blob, 104)
    sanity_ids = struct.unpack_from("<8H", blob, 120)

    if count > MAX_TENSORS or count == 0:
        raise ValueError(f"tensor_count {count}")
    if quant_bits != 8:
        raise ValueError(f"quant_bits {quant_bits}")
    if row_align != 16:
        raise ValueError(f"row_align {row_align}")
    if ctx != seq:
        raise ValueError(f"ctx {ctx} != seq {seq}")

    tensors = []
    for i in range(count):
        base = 256 + i * ENTRY
        tid, tflags, rows, cols, stride, doff, dbytes, soff = \
            struct.unpack_from("<HHIIIIII", blob, base)
        sbytes = struct.unpack_from("<I", blob, base + 28)[0]
        tier = blob[base + 32]
        if tid not in TID:
            raise ValueError(f"unknown tensor id {tid}")
        if tflags & F_Q8:
            if stride % 16 or stride < cols:
                raise ValueError(f"{TID[tid]}: stride {stride} cols {cols}")
            if dbytes != rows * stride:
                raise ValueError(f"{TID[tid]}: data {dbytes} != {rows * stride}")
            if doff % 16:
                raise ValueError(f"{TID[tid]}: data offset {doff} not 16-aligned")
            if soff != NO_SCALE:
                if soff % 4 or sbytes != rows * 4:
                    raise ValueError(f"{TID[tid]}: bad scale range")
        elif tflags & F_F32:
            if dbytes != rows * cols * 4 or doff % 4:
                raise ValueError(f"{TID[tid]}: bad f32 range")
        else:
            raise ValueError(f"{TID[tid]}: neither q8 nor f32")
        if doff + dbytes > payload_bytes:
            raise ValueError(f"{TID[tid]}: data past payload")
        if soff != NO_SCALE and soff + sbytes > payload_bytes:
            raise ValueError(f"{TID[tid]}: scales past payload")
        tensors.append(dict(tid=tid, flags=tflags, rows=rows, cols=cols,
                            stride=stride, doff=doff, dbytes=dbytes,
                            soff=soff, sbytes=sbytes, tier=tier))
    if tok_off % 4 or tok_off + tok_bytes > payload_bytes:
        raise ValueError("tokenizer range")

    return dict(flags=flags, dim=dim, hidden=hidden, nl=nl, nh=nh, nkvh=nkvh,
                vocab=vocab, seq=seq, ctx=ctx, tensors=tensors,
                q_bytes=q_bytes, s_bytes=s_bytes, n_bytes=n_bytes,
                tok_off=tok_off, tok_bytes=tok_bytes,
                sanity_len=sanity_len, sanity_topk=sanity_topk,
                sanity_crc=sanity_crc, sanity_prompt=sanity_prompt,
                sanity_ids=sanity_ids, payload=payload)


def dequant(hdr: dict, blob: bytes, t: dict) -> np.ndarray:
    payload = blob[HEADER_BYTES:]
    q = np.frombuffer(payload, dtype=np.int8, count=t["rows"] * t["stride"],
                      offset=t["doff"]).reshape(t["rows"], t["stride"])
    sc = np.frombuffer(payload, dtype="<f4", count=t["rows"],
                       offset=t["soff"]).astype(np.float32)
    out = q[:, :t["cols"]].astype(np.float32) * sc[:, None]
    return out


def build_legacy(hdr: dict, blob: bytes) -> tuple[bytes, dict[str, np.ndarray]]:
    by_id = {t["tid"]: t for t in hdr["tensors"]}
    parts = [struct.pack("<7i", hdr["dim"], hdr["hidden"], hdr["nl"],
                         hdr["nh"], hdr["nkvh"], hdr["vocab"], hdr["seq"])]
    views: dict[str, np.ndarray] = {}
    order = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    # llama2.c v0 order is interleaved per tensor, not 1..11 as ids.
    layout = [
        ("tok_embeddings", 1),
        ("rms_att", 9),
        ("wq", 2), ("wk", 3), ("wv", 4), ("wo", 5),
        ("rms_ffn", 10),
        ("w1", 6), ("w2", 7), ("w3", 8),
        ("rms_final", 11),
    ]
    for name, tid in layout:
        t = by_id[tid]
        if tid <= 8:
            arr = dequant(hdr, blob, t).reshape(-1)
        else:
            payload = blob[HEADER_BYTES:]
            arr = np.frombuffer(payload, dtype="<f4", count=t["rows"] * t["cols"],
                                offset=t["doff"]).astype(np.float32).reshape(-1)
        views[name] = arr
        parts.append(arr.astype("<f4").tobytes())
    # freqs_cos / freqs_sin are rebuilt by the firmware, so zeros are fine.
    hs = hdr["dim"] // hdr["nh"]
    nf = hdr["seq"] * hs // 2
    parts.append(np.zeros(nf, dtype="<f4").tobytes())
    parts.append(np.zeros(nf, dtype="<f4").tobytes())
    return b"".join(parts), views


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blob", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--golden", default="")
    ap.add_argument("--source", default="", help="optional fp32 checkpoint for error-bound check")
    ap.add_argument("--n-gen", type=int, default=12)
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    blob = Path(args.blob).read_bytes()
    try:
        hdr = parse_blob(blob)
    except ValueError as e:
        print(f"REJECT: {e}", file=sys.stderr)
        return 1
    print(f"blob {args.blob} ({len(blob)} B)")
    print(f"  sha256 {hashlib.sha256(blob).hexdigest()}")
    print(f"  cfg dim={hdr['dim']} hidden={hdr['hidden']} layers={hdr['nl']} "
          f"heads={hdr['nh']} kv_heads={hdr['nkvh']} vocab={hdr['vocab']} seq={hdr['seq']}")
    print(f"  payload={len(hdr['payload'])} B q={hdr['q_bytes']} "
          f"scales={hdr['s_bytes']} norms={hdr['n_bytes']} tokenizer={hdr['tok_bytes']}")
    print(f"  tokenizer range {hdr['tok_off']}..{hdr['tok_off'] + hdr['tok_bytes']}")
    print(f"  sanity prompt={list(hdr['sanity_prompt'][:hdr['sanity_len']])} "
          f"top{hdr['sanity_topk']} crc=0x{hdr['sanity_crc']:08x} "
          f"ids={list(hdr['sanity_ids'])}")
    print("  header CRC, payload SHA-256, tensor bounds: OK")

    # Tokenizer is embedded verbatim; compare with the source file.
    tok = Path(args.tokenizer).read_bytes()
    emb = hdr["payload"][hdr["tok_off"]:hdr["tok_off"] + hdr["tok_bytes"]]
    if emb != tok:
        print("TOKENIZER MISMATCH", file=sys.stderr)
        return 1
    print("  embedded tokenizer is byte-identical to the source: OK")

    legacy, views = build_legacy(hdr, blob)
    print(f"  rebuilt legacy v0 model: {len(legacy)} B")

    ref = load_module(here / "ref_llama.py", "ref_llama")

    # Quantisation error bound: |dequant - src| <= 0.5 * scale (+ clamp cases).
    if args.source:
        src_raw = Path(args.source).read_bytes()
        src_model = ref.load_model(Path(args.source))
        by_id = {t["tid"]: t for t in hdr["tensors"]}
        names = {1: "tok_embeddings", 2: "wq", 3: "wk", 4: "wv",
                 5: "wo", 6: "w1", 7: "w2", 8: "w3"}
        worst_ratio = 0.0
        for tid, name in names.items():
            t = by_id[tid]
            off, n = src_model.w[name]
            src = np.frombuffer(src_raw, dtype="<f4", count=n, offset=off)
            src = src.astype(np.float32).reshape(t["rows"], t["cols"])
            deq = dequant(hdr, blob, t)
            sc = np.frombuffer(blob, dtype="<f4", count=t["rows"],
                               offset=HEADER_BYTES + t["soff"]).astype(np.float32)
            err = np.abs(deq - src)
            bound = 0.5 * sc[:, None] + 1e-6 * np.abs(src)
            ratio = float((err / bound).max())
            worst_ratio = max(worst_ratio, ratio)
            print(f"  {name}: max|err|={float(err.max()):.3e} "
                  f"max step={float(sc.max()):.3e} bound_ratio={ratio:.4f}")
        if worst_ratio > 1.0001:
            print("QUANTISATION ERROR BOUND VIOLATED", file=sys.stderr)
            return 1
        print("  dequantised tensors are within half a quantisation step: OK")

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        f.write(legacy)
        tmp_path = Path(f.name)
    try:
        p = ref.load_model(tmp_path)
        runner = ref.Runner(p)
        vocab, scores, max_tok = ref.load_tokenizer(Path(args.tokenizer), p.vocab_size)
        prompt = "Once upon a time"
        toks = ref.encode(prompt, vocab, scores, max_tok)
        print(f"  prompt tokens: {toks}")
        if toks != [1, 403, 407, 261, 378]:
            print(f"WARNING: prompt tokens differ from the golden run", file=sys.stderr)
        pos = 0
        logits = None
        for t in toks:
            logits = runner.forward(t, pos)
            pos += 1
        first = sorted(range(len(logits)), key=lambda i: -logits[i])[:5]
        print(f"  blobby top5: {[(i, round(logits[i], 4)) for i in first]}")
        gen = []
        for _ in range(args.n_gen):
            nxt = max(range(len(logits)), key=lambda i: logits[i])
            gen.append(nxt)
            logits = runner.forward(nxt, pos)
            pos += 1
        print(f"  blobby greedy ids: {gen}")
    finally:
        tmp_path.unlink(missing_ok=True)

    if args.golden:
        ld = load_module(here / "logit_dump.py", "logit_dump")
        d = ld.parse(Path(args.golden))
        fp32 = d["fp32"][0]
        i8 = d["int8"][0]
        gold_top = ld.topk(fp32, 5)
        print(f"  device fp32 top5:  {[(i, round(float(fp32[i]), 4)) for i in gold_top]}")
        print(f"  device int8 top1={ld.topk(i8, 1)[0]} fp32 top1={gold_top[0]}")
        gold_ids = [int(t) for t in
                    [line.split()[-1] for line in
                     Path(args.golden).read_text(errors="replace").splitlines()
                     if line.startswith("TOK int8 0 ")]]
        print(f"  device greedy ids: {gold_ids}")
        ok_top = first == gold_top[:5] or first[0] == gold_top[0]
        ok_gen = gen == gold_ids[:len(gen)]
        print(f"  top1 match: {first[0] == gold_top[0]}  "
              f"top5 match: {first == gold_top}  greedy match: {ok_gen}")
        if not (first[0] == gold_top[0] and ok_gen):
            print("REFERENCE MISMATCH", file=sys.stderr)
            return 1

    print("VERIFY OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
