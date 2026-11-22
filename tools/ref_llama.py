#!/usr/bin/env python3
"""Pure-Python reference for the legacy llama2.c (v0) format.

No numpy/torch: the TinyStories-260K model is small enough that plain lists
finish a short generation in seconds. Used to validate the ESP32-S3 port and
to emit the exact greedy token ids the firmware must reproduce.

Layout (legacy v0, positive vocab_size => shared classifier):
  header: 7 x int32  dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab, seq_len
  tok_embeddings      vocab*dim
  per layer (all layers for one tensor, then next tensor):
      rms_att_weight  dim
      wq              dim*dim
      wk              dim*kv_dim
      wv              dim*kv_dim
      wo              dim*dim
      rms_ffn_weight  dim
      w1              dim*hidden
      w2              hidden*dim
      w3              dim*hidden
  rms_final           dim
  freqs_cos           seq_len*head_size/2   (skipped)
  freqs_sin           seq_len*head_size/2   (skipped)
  wcls                == tok_embeddings when shared
"""
from __future__ import annotations

import json
import math
import struct
import sys
from pathlib import Path

MODEL = Path(__file__).resolve().parent.parent / "model" / "stories260K.bin"
TOK = Path(__file__).resolve().parent.parent / "model" / "tok512.bin"


class Config:
    pass


def load_model(path: Path):
    raw = path.read_bytes()
    dim, hidden, nl, nh, nkvh, vocab, seq = struct.unpack("<7i", raw[:28])
    p = Config()
    p.dim, p.hidden_dim, p.n_layers = dim, hidden, nl
    p.n_heads, p.n_kv_heads, p.vocab_size, p.seq_len = nh, nkvh, vocab, seq
    p.kv_dim = dim * nkvh // nh
    p.head_size = dim // nh

    off = 28
    total = 4 * len(raw)

    def take(n):
        nonlocal off
        start = off
        off += 4 * n
        if off > len(raw):
            raise ValueError("model truncated")
        return start

    w = {}
    w["tok_embeddings"] = take(vocab * dim)
    w["rms_att"] = take(nl * dim)
    w["wq"] = take(nl * dim * dim)
    w["wk"] = take(nl * dim * p.kv_dim)
    w["wv"] = take(nl * dim * p.kv_dim)
    w["wo"] = take(nl * dim * dim)
    w["rms_ffn"] = take(nl * dim)
    w["w1"] = take(nl * dim * hidden)
    w["w2"] = take(nl * hidden * dim)
    w["w3"] = take(nl * dim * hidden)
    w["rms_final"] = take(dim)
    w["freqs_cos"] = take(seq * p.head_size // 2)
    w["freqs_sin"] = take(seq * p.head_size // 2)
    if off != len(raw):
        raise ValueError(f"layout mismatch: consumed {off} of {len(raw)} bytes")

    data = raw

    def f32(byte_off, n):
        return struct.unpack_from("<%df" % n, data, byte_off)

    p.w = {k: (v, n) for k, (v, n) in {
        "tok_embeddings": (w["tok_embeddings"], vocab * dim),
        "rms_att": (w["rms_att"], nl * dim),
        "wq": (w["wq"], nl * dim * dim),
        "wk": (w["wk"], nl * dim * p.kv_dim),
        "wv": (w["wv"], nl * dim * p.kv_dim),
        "wo": (w["wo"], nl * dim * dim),
        "rms_ffn": (w["rms_ffn"], nl * dim),
        "w1": (w["w1"], nl * dim * hidden),
        "w2": (w["w2"], nl * hidden * dim),
        "w3": (w["w3"], nl * dim * hidden),
        "rms_final": (w["rms_final"], dim),
    }.items()}
    p.f32 = f32
    return p


def tensor(p, name):
    off, n = p.w[name]
    return p.f32(off, n)


def rmsnorm(out, x, weight):
    n = len(x)
    ss = sum(v * v for v in x) / n
    scale = 1.0 / math.sqrt(ss + 1e-5)
    for i in range(n):
        out[i] = weight[i] * (x[i] * scale)


def matmul(out, x, w, n_in, n_out):
    """W (n_out, n_in) @ x (n_in,) -> out (n_out,), same arg order as run.c."""
    for i in range(n_out):
        base = i * n_in
        s = 0.0
        for j in range(n_in):
            s += w[base + j] * x[j]
        out[i] = s


def softmax(v):
    m = max(v)
    e = [math.exp(x - m) for x in v]
    s = sum(e)
    return [x / s for x in e]


class Runner:
    def __init__(self, p):
        self.p = p
        self.tok_emb = tensor(p, "tok_embeddings")
        self.rms_att = tensor(p, "rms_att")
        self.wq = tensor(p, "wq")
        self.wk = tensor(p, "wk")
        self.wv = tensor(p, "wv")
        self.wo = tensor(p, "wo")
        self.rms_ffn = tensor(p, "rms_ffn")
        self.w1 = tensor(p, "w1")
        self.w2 = tensor(p, "w2")
        self.w3 = tensor(p, "w3")
        self.rms_final = tensor(p, "rms_final")
        self.key_cache = [0.0] * (p.n_layers * p.seq_len * p.kv_dim)
        self.value_cache = [0.0] * (p.n_layers * p.seq_len * p.kv_dim)
        self.x = [0.0] * p.dim
        self.xb = [0.0] * p.dim
        self.xb2 = [0.0] * p.dim
        self.q = [0.0] * p.dim
        self.hb = [0.0] * p.hidden_dim
        self.hb2 = [0.0] * p.hidden_dim
        self.att = [0.0] * p.seq_len

    def forward(self, token, pos):
        p = self.p
        dim, kv_dim = p.dim, p.kv_dim
        hidden, head_size = p.hidden_dim, p.head_size
        kv_mul = p.n_heads // p.n_kv_heads
        x = self.x
        xb = self.xb

        x[:] = self.tok_emb[token * dim:(token + 1) * dim]

        for l in range(p.n_layers):
            rmsnorm(xb, x, self.rms_att[l * dim:(l + 1) * dim])
            loff = l * p.seq_len * kv_dim
            koff = loff + pos * kv_dim
            q = [0.0] * dim
            k = [0.0] * kv_dim
            v = [0.0] * kv_dim
            matmul(q, xb, self.wq[l * dim * dim:(l + 1) * dim * dim], dim, dim)
            matmul(k, xb, self.wk[l * dim * kv_dim:(l + 1) * dim * kv_dim], dim, kv_dim)
            matmul(v, xb, self.wv[l * dim * kv_dim:(l + 1) * dim * kv_dim], dim, kv_dim)
            # NOTE: RoPE must be applied *before* the key reaches the cache.
            # run.c rotates the row in place inside key_cache; copying first
            # would leave unrotated keys behind (that was a real bug here).
            for i in range(0, dim, 2):
                head_dim = i % head_size
                freq = 1.0 / (10000.0 ** (head_dim / head_size))
                val = pos * freq
                fcr, fci = math.cos(val), math.sin(val)
                rotn = 2 if i < kv_dim else 1
                for vi in range(rotn):
                    vec = q if vi == 0 else k
                    v0, v1 = vec[i], vec[i + 1]
                    vec[i] = v0 * fcr - v1 * fci
                    vec[i + 1] = v0 * fci + v1 * fcr

            self.key_cache[koff:koff + kv_dim] = k
            self.value_cache[koff:koff + kv_dim] = v

            for h in range(p.n_heads):
                qh = q[h * head_size:(h + 1) * head_size]
                scores = []
                for t in range(pos + 1):
                    kb = loff + t * kv_dim + (h // kv_mul) * head_size
                    kh = self.key_cache[kb:kb + head_size]
                    scores.append(sum(a * b for a, b in zip(qh, kh)) / math.sqrt(head_size))
                probs = softmax(scores)
                acc = [0.0] * head_size
                for t in range(pos + 1):
                    vb = loff + t * kv_dim + (h // kv_mul) * head_size
                    vh = self.value_cache[vb:vb + head_size]
                    a = probs[t]
                    for i in range(head_size):
                        acc[i] += a * vh[i]
                xb[h * head_size:(h + 1) * head_size] = acc

            xb2 = [0.0] * dim
            matmul(xb2, xb, self.wo[l * dim * dim:(l + 1) * dim * dim], dim, dim)
            for i in range(dim):
                x[i] += xb2[i]

            rmsnorm(xb, x, self.rms_ffn[l * dim:(l + 1) * dim])
            hb = [0.0] * hidden
            hb2 = [0.0] * hidden
            matmul(hb, xb, self.w1[l * dim * hidden:(l + 1) * dim * hidden], dim, hidden)
            matmul(hb2, xb, self.w3[l * dim * hidden:(l + 1) * dim * hidden], dim, hidden)
            for i in range(hidden):
                s = 1.0 / (1.0 + math.exp(-hb[i]))
                hb[i] = hb[i] * s * hb2[i]
            xb2 = [0.0] * dim
            matmul(xb2, hb, self.w2[l * hidden * dim:(l + 1) * hidden * dim], hidden, dim)
            for i in range(dim):
                x[i] += xb2[i]

        rmsnorm(xb, x, self.rms_final)
        logits = [0.0] * p.vocab_size
        matmul(logits, xb, self.tok_emb, dim, p.vocab_size)
        return logits


def load_tokenizer(path: Path, vocab_size: int):
    """llama2.c tokenizer.bin: int32 max_token_length, then vocab_size entries."""
    raw = path.read_bytes()
    (max_token_length,) = struct.unpack_from("<i", raw, 0)
    off = 4
    vocab = []
    scores = []
    for _ in range(vocab_size):
        (score,) = struct.unpack_from("<f", raw, off)
        off += 4
        (ln,) = struct.unpack_from("<i", raw, off)
        off += 4
        vocab.append(raw[off:off + ln])
        scores.append(score)
        off += ln
    return vocab, scores, max_token_length


def encode(text: str, vocab, scores, max_token_length):
    """Exact port of run.c encode(): byte fallback (+3) then score-ordered merges."""
    lookup = {vocab[i]: i for i in range(len(vocab))}
    tokens = [1]  # BOS
    if text:
        tokens.append(lookup[b" "])
    buf = text.encode("utf-8")
    i = 0
    while i < len(buf):
        # gather one UTF-8 codepoint
        start = i
        i += 1
        while i < len(buf) and (buf[i] & 0xC0) == 0x80 and (i - start) < 4:
            i += 1
        piece = buf[start:i]
        idx = lookup.get(piece)
        if idx is not None:
            tokens.append(idx)
        else:
            tokens.extend(b + 3 for b in piece)
    # merge the best consecutive pair each iteration
    while True:
        best_score, best_id, best_idx = -1e10, -1, -1
        for j in range(len(tokens) - 1):
            cand = vocab[tokens[j]] + vocab[tokens[j + 1]]
            cid = lookup.get(cand)
            if cid is not None and scores[cid] > best_score:
                best_score, best_id, best_idx = scores[cid], cid, j
        if best_idx == -1:
            break
        tokens[best_idx] = best_id
        del tokens[best_idx + 1]
    return tokens


def decode(tokens, vocab):
    out = b""
    for t in tokens:
        if not (0 <= t < len(vocab)):
            continue
        piece = vocab[t]
        if len(piece) == 6 and piece.startswith(b"<0x") and piece.endswith(b">"):
            try:
                out += bytes([int(piece[3:5], 16)])
                continue
            except ValueError:
                pass
        out += piece
    return out.decode("utf-8", errors="replace")


def main():
    p = load_model(MODEL)
    vocab, scores, max_token_length = load_tokenizer(TOK, p.vocab_size)
    print(f"cfg dim={p.dim} hidden={p.hidden_dim} layers={p.n_layers} "
          f"heads={p.n_heads} kv_heads={p.n_kv_heads} vocab={p.vocab_size} seq={p.seq_len}")
    print(f"vocab tokens={len(vocab)} max_token_length={max_token_length}")

    prompt = sys.argv[1] if len(sys.argv) > 1 else "Once upon a time"
    n_gen = int(sys.argv[2]) if len(sys.argv) > 2 else 24

    runner = Runner(p)
    tokens = encode(prompt, vocab, scores, max_token_length)
    print("prompt tokens:", tokens)

    pos = 0
    for t in tokens:
        logits = runner.forward(t, pos)
        pos += 1

    first_top = sorted(range(len(logits)), key=lambda i: -logits[i])[:5]
    print("first top5:", [(i, round(logits[i], 4)) for i in first_top])

    generated = []
    for _ in range(n_gen):
        nxt = max(range(len(logits)), key=lambda i: logits[i])
        generated.append(nxt)
        logits = runner.forward(nxt, pos)
        pos += 1

    print("generated ids:", generated)
    print("generated text:", repr(decode(generated, vocab)))
    print("full text:", repr(decode(tokens + generated, vocab)))

    out = {
        "prompt": prompt,
        "prompt_tokens": tokens,
        "generated_ids": generated,
        "text": decode(generated, vocab),
    }
    dest = Path(__file__).resolve().parent.parent / "expected.json"
    dest.write_text(json.dumps(out, indent=2))
    print("wrote", dest)


if __name__ == "__main__":
    main()
