#!/usr/bin/env python3
"""Turn a bench UART log into the tables the paper needs.

Usage: analyze_bench.py <bench.log> [ref_logits.json] [out.json]
"""
import json
import math
import statistics
import struct
import sys
from pathlib import Path


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - k) + xs[hi] * (k - lo)


def parse(path):
    tims, runs, ttfts, mems = [], [], [], []
    cfg = prompt = None
    logits = {}
    cur = None
    for raw in Path(path).read_text(errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("LOGD "):
            _, label, step, n = line.split()
            cur = (label, int(step), int(n), [])
            continue
        if line.startswith("LOGE "):
            if cur:
                logits[(cur[0], cur[1])] = cur[3]
            cur = None
            continue
        if cur is not None:
            for h in line.split():
                if len(h) == 8 and all(c in "0123456789abcdefABCDEF" for c in h):
                    cur[3].append(struct.unpack("<f", struct.pack("<I", int(h, 16)))[0])
            continue
        if line.startswith("TIM "):
            _, label, rep, pos, us = line.split()
            tims.append((label, int(rep), int(pos), int(us)))
        elif line.startswith("RUN "):
            _, label, rep, np_, ng, total, gen = line.split()
            runs.append((label, int(rep), int(np_), int(ng), int(total), int(gen)))
        elif line.startswith("TTFT "):
            _, label, rep, np_, us = line.split()
            ttfts.append((label, int(rep), int(np_), int(us)))
        elif line.startswith("MEM "):
            mems.append(line)
        elif line.startswith("CFG "):
            cfg = line
        elif line.startswith("PROMPT "):
            prompt = line
    return dict(tims=tims, runs=runs, ttfts=ttfts, mems=mems, cfg=cfg,
                prompt=prompt, logits=logits)


def summarize_var(d):
    out = {}
    for label in sorted({r[0] for r in d["runs"]}):
        rows = [r for r in d["runs"] if r[0] == label and r[5] > 0]
        if not rows:
            continue
        tps = [r[3] * 1e6 / r[5] for r in rows]
        out[label] = dict(
            n=len(tps),
            median=statistics.median(tps),
            p25=pct(tps, 0.25),
            p75=pct(tps, 0.75),
            min=min(tps),
            max=max(tps),
            runs=tps,
        )
    return out


def summarize_ctx(d):
    by_rep = {}
    for label, rep, pos, us in d["tims"]:
        if label != "ctx" or us <= 0:
            continue
        by_rep.setdefault(rep, []).append((pos, us))
    if not by_rep:
        return {}

    buckets = [(0, 16), (16, 32), (32, 64), (64, 128),
               (128, 256), (256, 384), (384, 512)]
    res = []
    for lo, hi in buckets:
        vals = [1e6 / us for rows in by_rep.values() for pos, us in rows
                if lo <= pos < hi]
        if vals:
            res.append(dict(lo=lo, hi=hi, median=statistics.median(vals),
                            n=len(vals)))

    short = [1e6 / us for rows in by_rep.values() for pos, us in rows
             if 5 <= pos <= 16]

    by_pos = {}
    for pos in range(max(max(p for p, _ in rows) for rows in by_rep.values()) + 1):
        vals = [1e6 / us for rows in by_rep.values() for p, us in rows if p == pos]
        if vals:
            by_pos[pos] = statistics.median(vals)

    return dict(buckets=res,
                short_median=statistics.median(short) if short else None,
                short_n=len(short),
                by_pos=by_pos)


def logit_metrics(d, ref):
    out = {}
    for label in ("int8", "fp32"):
        steps = []
        for step in range(len(ref["logits"])):
            a = d["logits"].get((label, step))
            if not a:
                continue
            b = ref["logits"][step]
            n = min(len(a), len(b))
            a, b = a[:n], b[:n]
            diffs = [abs(x - y) for x, y in zip(a, b)]
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(y * y for y in b))
            cos = dot / (na * nb) if na and nb else float("nan")
            ia = max(range(n), key=lambda i: a[i])
            ib = max(range(n), key=lambda i: b[i])
            sa = sorted(range(n), key=lambda i: -a[i])[:5]
            sb = sorted(range(n), key=lambda i: -b[i])[:5]
            ma = a[ia] - sorted(a)[-2] if n > 1 else float("nan")
            mb = b[ib] - sorted(b)[-2] if n > 1 else float("nan")
            steps.append(dict(
                step=step,
                max_abs=max(diffs),
                mean_abs=sum(diffs) / n,
                cosine=cos,
                top1_match=(ia == ib),
                top5_overlap=len(set(sa) & set(sb)),
                margin_dev=ma - mb,
            ))
        out[label] = steps
    return out


def main():
    log = Path(sys.argv[1] if len(sys.argv) > 1 else "logs/bench_h16.log")
    ref_path = Path(sys.argv[2] if len(sys.argv) > 2 else "logs/ref_logits.json")
    d = parse(log)

    print(d["cfg"] or "(no CFG line)")
    print(d["prompt"] or "(no PROMPT line)")

    print("\n-- memory --")
    for m in d["mems"]:
        print(m)

    print("\n-- variance (short runs) --")
    var = summarize_var(d)
    for label, s in var.items():
        print(f"{label}: n={s['n']} median={s['median']:.2f} tok/s "
              f"IQR=[{s['p25']:.2f},{s['p75']:.2f}] "
              f"min={s['min']:.2f} max={s['max']:.2f}")

    print("\n-- context --")
    ctx = summarize_ctx(d)
    if ctx:
        print(f"short (pos 5-16) median={ctx['short_median']:.2f} tok/s "
              f"over {ctx['short_n']} samples")
        for b in ctx["buckets"]:
            print(f"pos {b['lo']:3d}-{b['hi'] - 1:3d}: "
                  f"{b['median']:7.2f} tok/s (n={b['n']})")

    lm = None
    if ref_path.exists():
        ref = json.loads(ref_path.read_text())
        print("\n-- logits --")
        lm = logit_metrics(d, ref)
        for label, steps in lm.items():
            if not steps:
                continue
            print(f"{label}: steps={len(steps)} "
                  f"max_abs(max)={max(s['max_abs'] for s in steps):.4g} "
                  f"mean_abs(avg)={sum(s['mean_abs'] for s in steps) / len(steps):.4g} "
                  f"cos(min)={min(s['cosine'] for s in steps):.6f} "
                  f"top1={sum(s['top1_match'] for s in steps)}/{len(steps)} "
                  f"top5={sum(s['top5_overlap'] for s in steps) / len(steps):.2f}/5")
    else:
        print(f"\n(no reference logits at {ref_path})")

    out = Path(sys.argv[3] if len(sys.argv) > 3 else log.with_suffix(".summary.json"))
    out.write_text(json.dumps(dict(cfg=d["cfg"], var=var, ctx=ctx,
                                   mems=d["mems"], logits=lm), indent=2))
    print("\nwrote", out)


if __name__ == "__main__":
    main()
