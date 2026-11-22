#!/usr/bin/env python3
"""Dump reference logits from the validated pure-Python llama2.c port.

The device prints its own logits for the same 12 steps; this file is the
ground truth for the logit-level agreement table.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ref_llama import MODEL, TOK, Runner, encode, load_model, load_tokenizer  # noqa: E402


def main():
    n_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else (
        Path(__file__).resolve().parent.parent / "logs" / "ref_logits.json"
    )
    prompt = "Once upon a time"

    p = load_model(MODEL)
    vocab, scores, max_token_length = load_tokenizer(TOK, p.vocab_size)
    tokens = encode(prompt, vocab, scores, max_token_length)

    runner = Runner(p)
    pos = 0
    logits = None
    for t in tokens:
        logits = runner.forward(t, pos)
        pos += 1

    steps = []
    ids = []
    for _ in range(n_steps):
        steps.append(logits)
        nxt = max(range(len(logits)), key=lambda i: logits[i])
        ids.append(nxt)
        logits = runner.forward(nxt, pos)
        pos += 1

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "prompt": prompt,
        "prompt_tokens": tokens,
        "generated_ids": ids,
        "logits": steps,
    }))
    print(f"wrote {out}")
    print("prompt tokens:", tokens)
    print("generated ids:", ids)


if __name__ == "__main__":
    main()
