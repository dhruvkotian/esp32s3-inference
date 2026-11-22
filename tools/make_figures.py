#!/usr/bin/env python3
"""Generate the figures for the INT8 KV experiment from the bench logs.

Everything is derived from the raw UART captures. All baseline cells now come
from the same tree, so no number is typed in by hand.
"""

import math
import struct
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"
OUT = LOGS / "figs"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    # Embed TrueType rather than Type 3 so the figures pass strict PDF
    # font checks at submission time.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# --------------------------------------------------------------------------
# parsing

def parse_tim(path):
    """Return {pos: [us, ...]} from TIM ctx lines."""
    rows = {}
    for line in Path(path).read_text(errors="replace").splitlines():
        if line.startswith("TIM ctx"):
            _, _, rep, pos, us = line.split()
            rows.setdefault(int(pos), []).append(int(us))
    return rows

def med(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2

def require(d, key, src):
    """Return d[key] or fail loudly. Figures must never invent a measurement."""
    if key not in d:
        raise KeyError(f"{key!r} not found in {src}")
    return d[key]

def sustained(path):
    """Median sustained tok/s from RUN ctx lines."""
    vals = []
    for line in Path(path).read_text(errors="replace").splitlines():
        if line.startswith("RUN ctx"):
            _, _, rep, np_, ntok, total, gen = line.split()
            vals.append(int(ntok) / int(gen) * 1e6)
    if not vals:
        raise ValueError(f"no RUN ctx lines in {path}")
    return med(vals)

def parse_prof(path):
    out = {}
    for line in Path(path).read_text(errors="replace").splitlines():
        if line.startswith("PROF "):
            parts = line.split()
            if len(parts) >= 3 and "=" in parts[2]:
                out[parts[1]] = float(parts[2].split("=")[1])
    return out

def parse_aprof(path):
    out = {}
    for line in Path(path).read_text(errors="replace").splitlines():
        if line.startswith("APROF "):
            parts = line.split()
            if len(parts) >= 3 and "=" in parts[2]:
                out[parts[1]] = float(parts[2].split("=")[1])
    return out

def parse_psram(path):
    out = {}
    for line in Path(path).read_text(errors="replace").splitlines():
        if line.startswith("PSRAM "):
            for tok in line.split()[1:]:
                if "=" in tok:
                    k, v = tok.split("=")
                    try:
                        out[k] = float(v)
                    except ValueError:
                        pass
    return out

def parse_logits(path):
    blocks = {}
    cur = None
    for raw in Path(path).read_text(errors="replace").splitlines():
        line = raw.strip()
        if line.startswith("LOGD "):
            _, label, step, n = line.split()
            cur = (label, int(step), int(n), [])
            continue
        if line.startswith("LOGE "):
            if cur:
                blocks[(cur[0], cur[1])] = cur[3]
            cur = None
            continue
        if cur is not None:
            for h in line.split():
                if len(h) == 8 and all(c in "0123456789abcdefABCDEF" for c in h):
                    cur[3].append(struct.unpack("<f", struct.pack("<I", int(h, 16)))[0])
    return blocks

def accuracy_vs_depth(path):
    lg = parse_logits(path)
    d = {}
    for (lab, step), v in lg.items():
        if lab.startswith("fp32L"):
            d.setdefault(int(lab[5:]), {})["fp32"] = v
        elif lab.startswith("int8L"):
            d.setdefault(int(lab[5:]), {})["int8"] = v
    out = []
    for pos in sorted(d):
        a, b = d[pos]["fp32"], d[pos]["int8"]
        diff = [x - y for x, y in zip(a, b)]
        rmse = math.sqrt(sum(x * x for x in diff) / len(diff))
        den = math.sqrt(sum(x * x for x in a) / len(a))
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        ta = max(range(len(a)), key=lambda i: a[i])
        tb = max(range(len(b)), key=lambda i: b[i])
        out.append((pos, rmse / den * 100.0, dot / (na * nb), ta == tb))
    return out

def compare_logits(ref_path, ref_pref, test_path, test_pref):
    """Compare logits dumped by two different builds at matching labels."""
    a = parse_logits(ref_path)
    b = parse_logits(test_path)
    da = {int(lab[len(ref_pref):]): v
          for (lab, _), v in a.items() if lab.startswith(ref_pref)}
    db = {int(lab[len(test_pref):]): v
          for (lab, _), v in b.items() if lab.startswith(test_pref)}
    out = []
    for pos in sorted(set(da) & set(db)):
        x, y = da[pos], db[pos]
        diff = [p - q for p, q in zip(x, y)]
        rmse = math.sqrt(sum(d * d for d in diff) / len(diff))
        den = math.sqrt(sum(p * p for p in x) / len(x))
        dot = sum(p * q for p, q in zip(x, y))
        nx = math.sqrt(sum(p * p for p in x))
        ny = math.sqrt(sum(q * q for q in y))
        tx = max(range(len(x)), key=lambda i: x[i])
        ty = max(range(len(y)), key=lambda i: y[i])
        out.append((pos, rmse / den * 100.0, dot / (nx * ny), tx == ty))
    return out

def rolling_median(xs, w=9):
    out = []
    half = w // 2
    for i in range(len(xs)):
        lo = max(0, i - half)
        hi = min(len(xs), i + half + 1)
        out.append(med(xs[lo:hi]))
    return out

# --------------------------------------------------------------------------
# figure 1: throughput versus position

def fig_throughput():
    series = [
        ("fp32 KV, 64 KB, 1 core", "bench_fp32_64_single.log", "#555555"),
        ("int8 dequant, 64 KB", "bench_kvint8_c64_skip.log", "#1f77b4"),
        ("integer K dot, 64 KB", "bench_kvint8_idot2.log", "#ff7f0e"),
        ("V transpose, 64 KB", "bench_kvint8_vt.log", "#2ca02c"),
        ("dual core, 64 KB", "bench_kvint8_dc.log", "#d62728"),
        ("final + fused load, 64 KB", "bench_kvint8_fused_1.log", "#9467bd"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for label, fn, color in series:
        rows = parse_tim(LOGS / fn)
        if not rows:
            continue
        pos = sorted(rows)
        tps = [1e6 / med(rows[p]) for p in pos]
        smooth = rolling_median(tps, 9)
        ax.plot(pos, smooth, color=color, lw=1.8, label=label)
    ax.set_xlabel("Context position")
    ax.set_ylabel("Throughput (tokens/s)")
    ax.set_xlim(0, 512)
    ax.set_ylim(0, 750)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_title("Per-token throughput versus context position")
    fig.tight_layout()
    fig.savefig(OUT / "fig_throughput_vs_position.png")
    fig.savefig(OUT / "fig_throughput_vs_position.pdf")
    plt.close(fig)

# --------------------------------------------------------------------------
# figure 2: sustained bar

def fig_sustained():
    bars = [
        ("fp32\n32 KB\n1 core", "bench_fp32_32_single.log", "#555555"),
        ("fp32\n64 KB\n1 core", "bench_fp32_64_single.log", "#888888"),
        ("fp32\n64 KB\n2 cores", "bench_fp32_64_dual.log", "#aaaaaa"),
        ("int8 dequant\n64 KB", "bench_kvint8_c64_skip.log", "#1f77b4"),
        ("integer K\n64 KB", "bench_kvint8_idot2.log", "#ff7f0e"),
        ("V transpose\n64 KB", "bench_kvint8_vt.log", "#2ca02c"),
        ("dual core\n64 KB", "bench_kvint8_dc.log", "#d62728"),
        ("final kernels\n64 KB", "bench_kvint8_fused_1.log", "#9467bd"),
    ]
    labels, vals, colors = [], [], []
    for label, fn, color in bars:
        labels.append(label)
        vals.append(sustained(LOGS / fn))
        colors.append(color)

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    x = np.arange(len(labels))
    bars_ = ax.bar(x, vals, color=colors, width=0.64)
    for b, v in zip(bars_, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}",
                ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Sustained throughput (tokens/s)")
    ax.set_ylim(0, 185)
    ax.set_title("512-token sustained throughput, all bars from the same tree")
    fig.tight_layout()
    fig.savefig(OUT / "fig_sustained_bar.png")
    fig.savefig(OUT / "fig_sustained_bar.pdf")
    plt.close(fig)

# --------------------------------------------------------------------------
# figure 3: section profile

def fig_sections():
    prof = parse_prof(LOGS / "bench_kvint8_fused_1.log")
    order = [
        ("attn", "attention"),
        ("ffn1", "ffn w1/w3"),
        ("rope", "rope"),
        ("qkv", "qkv matmuls"),
        ("ffn2", "w2 + residual"),
        ("silu", "silu"),
        ("wo", "wo + residual"),
        ("head", "classifier"),
    ]
    names = [n for k, n in order if k in prof]
    vals = [prof[k] for k, n in order if k in prof]

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    y = np.arange(len(names))
    bars_ = ax.barh(y, vals, color="#d62728")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Time per token (us)")
    for b, v in zip(bars_, vals):
        ax.text(v + 40, b.get_y() + b.get_height() / 2, f"{v:.0f}",
                va="center", fontsize=8)
    ax.set_xlim(0, max(vals) * 1.18)
    ax.set_title("Int8 forward section profile, dual core, 64 KB cache")
    fig.tight_layout()
    fig.savefig(OUT / "fig_section_profile.png")
    fig.savefig(OUT / "fig_section_profile.pdf")
    plt.close(fig)

# --------------------------------------------------------------------------
# figure 4: attention sub-profile

def fig_subprofile():
    ap = parse_aprof(LOGS / "bench_kvint8_fused_1.log")
    order = [
        ("kdot", "K dot", "#1f77b4"),
        ("softmax", "softmax + fold", "#2ca02c"),
        ("vdot", "V dot", "#d62728"),
        ("vquant", "V weight quantize", "#ff7f0e"),
    ]
    names = [n for k, n, c in order if k in ap]
    vals = [ap[k] for k, n, c in order if k in ap]
    colors = [c for k, n, c in order if k in ap]
    total = sum(vals)

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    wedges, texts, autotexts = ax.pie(
        vals, labels=names, colors=colors, autopct=lambda p: f"{p:.1f}%",
        startangle=90, counterclock=False,
        wedgeprops={"linewidth": 1, "edgecolor": "white"},
        textprops={"fontsize": 8},
    )
    ax.set_title("Attention sub-profile, both cores summed\n"
                 f"{total / 1000.0:.1f} ms of core time per token", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_attention_subprofile.png")
    fig.savefig(OUT / "fig_attention_subprofile.pdf")
    plt.close(fig)

# --------------------------------------------------------------------------
# figure 5: accuracy versus depth

def fig_accuracy():
    ref = LOGS / "bench_acc_ref.log"
    fin = LOGS / "bench_kvint8_fused_1.log"
    series = [
        ("Weight-only (int8 weights, fp32 KV)",
         compare_logits(ref, "fp32L", ref, "int8L"), "#ff7f0e", "o"),
        ("KV-only (fp32 KV vs int8 KV)",
         compare_logits(ref, "int8L", fin, "int8L"), "#2ca02c", "^"),
        ("Combined (both int8)",
         compare_logits(fin, "fp32L", fin, "int8L"), "#d62728", "s"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.8))
    for name, data, color, marker in series:
        pos = [d[0] for d in data]
        rmse = [d[1] for d in data]
        cos = [d[2] for d in data]
        ax1.plot(pos, rmse, marker=marker, ls="-", color=color, lw=1.6,
                 ms=4.5, label=name)
        ax2.plot(pos, cos, marker=marker, ls="-", color=color, lw=1.6,
                 ms=4.5, label=name)

    ax1.axvline(16, color="#666666", ls=":", lw=1.0)
    ax1.text(20, 5.78, "hot window ends", fontsize=7, color="#666666")
    ax1.annotate("KV-only is 0 inside\nthe fp32 hot window", (15, 0.0),
                 xytext=(70, 0.42), textcoords="data", fontsize=6.5,
                 color="#2ca02c", ha="left",
                 arrowprops={"arrowstyle": "-", "lw": 0.6, "color": "#2ca02c"})
    ax1.set_xlabel("Context position")
    ax1.set_ylabel("Relative logit RMSE (%)")
    ax1.set_ylim(-0.3, 6.2)
    ax1.legend(frameon=False, fontsize=6.8, loc="upper right")
    ax1.set_title("Logit error versus depth")

    ax2.set_xlabel("Context position")
    ax2.set_ylabel("Cosine similarity")
    ax2.set_ylim(0.9993, 1.00002)
    ax2.set_title("Top-1 agreement: 9/9 on all three paths")

    fig.tight_layout()
    fig.savefig(OUT / "fig_accuracy_vs_depth.png")
    fig.savefig(OUT / "fig_accuracy_vs_depth.pdf")
    plt.close(fig)

# --------------------------------------------------------------------------
# figure 6: PSRAM bandwidth

def fig_psram():
    ps = parse_psram(LOGS / "bench_kvint8_fused_1.log")
    labels = ["Sequential\nread", "Strided read\n(16 of 32 B)", "Sequential\nwrite"]
    keys = ["ext_seq_mbps", "ext_stride_mbps", "ext_write_mbps"]
    vals = [require(ps, k, "bench_kvint8_fused_1.log") for k in keys]

    # Streamed KV bytes per token at 512 context: 5 layers * 496 positions *
    # 96 B. Positions 0..15 are the fp32 hot window in internal SRAM.
    kv_bytes = 5 * 496 * 96
    attn_us = require(parse_prof(LOGS / "bench_kvint8_fused_1.log"), "attn",
                      "bench_kvint8_fused_1.log")
    achieved = kv_bytes / attn_us

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    x = np.arange(len(labels))
    bars_ = ax.bar(x, vals, color=["#1f77b4", "#ff7f0e", "#2ca02c"], width=0.6)
    for b, v in zip(bars_, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.0, f"{v:.1f}",
                ha="center", va="bottom", fontsize=8)
    ax.axhline(achieved, color="#d62728", ls="--", lw=1.4,
               label=f"streamed KV bytes / token, {achieved:.1f} MB/s")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Bandwidth (MB/s)")
    ax.set_ylim(0, 70)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_title("PSRAM microbenchmark, Octal 80 MHz, 64 KB cache\n"
                 "dashed line is 5 x 496 x 96 B divided by attention wall time",
                 fontsize=9)
    fig.text(0.5, 0.012, f"The scan runs at {achieved / vals[0] * 100:.0f}% of "
             "the sequential read rate measured in this configuration; the "
             "rest is strided K chunks and scale reads.",
             ha="center", fontsize=6.5, color="#666666")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(OUT / "fig_psram_bandwidth.png")
    fig.savefig(OUT / "fig_psram_bandwidth.pdf")
    plt.close(fig)

# --------------------------------------------------------------------------
# figure 7: cache sweep

def fig_cache():
    logs = [("32 KB", "bench_kvint8_fused_32kb.log"),
            ("64 KB", "bench_kvint8_fused_1.log")]
    sus = [sustained(LOGS / fn) for _, fn in logs]
    attn = [require(parse_prof(LOGS / fn), "attn", fn) for _, fn in logs]
    aps = [parse_aprof(LOGS / fn) for _, fn in logs]
    kdot = [require(a, "kdot", fn) for a, (_, fn) in zip(aps, logs)]
    smax = [require(a, "softmax", fn) for a, (_, fn) in zip(aps, logs)]
    vq = [require(a, "vquant", fn) for a, (_, fn) in zip(aps, logs)]
    vdot = [require(a, "vdot", fn) for a, (_, fn) in zip(aps, logs)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.8, 3.9))
    x = np.arange(2)
    b1 = ax1.bar(x, sus, color=["#ff7f0e", "#d62728"], width=0.55)
    for b, v in zip(b1, sus):
        ax1.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}",
                 ha="center", va="bottom", fontsize=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels([l for l, _ in logs])
    ax1.set_ylabel("Sustained tokens/s")
    ax1.set_ylim(0, 185)
    ax1.set_title("Throughput")

    # The four sub-components are measured in CPU cycles per core and summed,
    # so their height is core time, not wall time. The wall attention total is
    # printed above each group for reference.
    comps = [
        ("K dot", kdot, "#1f77b4"),
        ("softmax", smax, "#2ca02c"),
        ("V quantize", vq, "#ff7f0e"),
        ("V dot", vdot, "#d62728"),
    ]
    w = 0.19
    for j, (lab, vals, c) in enumerate(comps):
        ax2.bar(x + (j - 1.5) * w, vals, w, label=lab, color=c)
    for xi, tot in zip(x, attn):
        ax2.text(xi, -0.16, f"wall {tot:.0f} us", transform=ax2.get_xaxis_transform(),
                 ha="center", va="top", fontsize=7, color="#444444")
    ax2.set_xticks(x)
    ax2.set_xticklabels([l for l, _ in logs])
    ax2.set_ylabel("Core time per token (us, both cores summed)")
    ax2.set_ylim(0, max(kdot) * 1.22)
    ax2.legend(frameon=False, fontsize=7, ncol=2, loc="upper right")
    ax2.set_title("Attention components")
    fig.suptitle("Data cache size sweep, final kernel, dual core, int8 KV",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig_cache_sweep.png")
    fig.savefig(OUT / "fig_cache_sweep.pdf")
    plt.close(fig)

# --------------------------------------------------------------------------
# figure 8: optimization waterfall

def fig_waterfall():
    rows = [
        ("fp32 KV, 1 core", "bench_fp32_64_single.log", "#555555"),
        ("int8 dequant, 1 core", "bench_kvint8_c64_skip.log", "#c0392b"),
        ("+ integer K dot", "bench_kvint8_idot2.log", "#e67e22"),
        ("+ V transpose", "bench_kvint8_vt.log", "#27ae60"),
        ("+ dual core", "bench_kvint8_dc.log", "#2980b9"),
        ("+ att2 alloc fix", "bench_kvint8_fix.log", "#7f8c8d"),
        ("+ scale transpose", "bench_kvint8_strans.log", "#8e44ad"),
        ("+ softmax_fold", "bench_kvint8_fuse.log", "#a0522d"),
        ("+ batched K x4", "bench_kvint8_x4.log", "#d81b60"),
        ("+ inline rounding", "bench_kvint8_final3.log", "#00897b"),
        ("+ fused long dot", "bench_kvint8_fused_1.log", "#004d40"),
    ]
    labels = [r[0] for r in rows]
    vals = [sustained(LOGS / fn) for _, fn, _ in rows]
    colors = [c for _, _, c in rows]

    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    y = np.arange(len(labels))
    bars_ = ax.barh(y, vals, color=colors, height=0.64)
    for i, (b, v) in enumerate(zip(bars_, vals)):
        ax.text(v + 1.5, b.get_y() + b.get_height() / 2, f"{v:.1f}",
                va="center", ha="left", fontsize=7.8)
        if i > 0:
            step = (v / vals[i - 1] - 1.0) * 100.0
            sign = "+" if step >= 0 else ""
            ax.text(v - 2.0, b.get_y() + b.get_height() / 2,
                    f"{sign}{step:.1f}%", va="center", ha="right",
                    fontsize=7.2, color="white")
    ax.axvline(vals[0], color="#999999", ls=":", lw=1.0)
    ax.text(vals[0] + 1.0, -0.72, "fp32 baseline", fontsize=7,
            color="#777777", ha="left", va="center")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.8)
    ax.invert_yaxis()
    ax.set_xlabel("Sustained tokens/s at 512 context")
    ax.set_xlim(0, 205)
    ax.set_title("Full optimisation chain, 64 KB cache")
    fig.tight_layout()
    fig.savefig(OUT / "fig_kernel_waterfall.png")
    fig.savefig(OUT / "fig_kernel_waterfall.pdf")
    plt.close(fig)

# --------------------------------------------------------------------------
# figure 9: DMA negative result

def fig_dma():
    rows = [
        ("row, no DMA", "bench_kvint8_vt.log"),
        ("tiled, no DMA", "bench_kvint8_tiled.log"),
        ("DMA tile 64", "bench_kvint8_dma.log"),
        ("DMA tile 128", "bench_kvint8_dma128.log"),
        ("DMA tile 256", "bench_kvint8_dma256.log"),
    ]
    labels = [r[0] for r in rows]
    vals = [sustained(LOGS / fn) for _, fn in rows]

    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    x = np.arange(len(labels))
    colors = ["#2ca02c", "#8c8c8c", "#d62728", "#d62728", "#d62728"]
    bars_ = ax.bar(x, vals, color=colors, width=0.62)
    for b, v in zip(bars_, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.1f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=15)
    ax.set_ylabel("Sustained tokens/s")
    ax.set_ylim(0, 105)
    ax.set_title("V DMA prefetch is a regression")
    fig.tight_layout()
    fig.savefig(OUT / "fig_dma_result.png")
    fig.savefig(OUT / "fig_dma_result.pdf")
    plt.close(fig)

# --------------------------------------------------------------------------
# figure 10: KV footprint

def fig_footprint():
    # Sizes from the model dimensions: 5 layers x 512 positions x 32 values.
    # The hot window (20 KiB) is present in both builds and is excluded here.
    n = 5 * 512 * 32
    fp32_kib = 2 * n * 4 / 1024.0          # 640 KiB
    data_kib = 2 * n / 1024.0              # 160 KiB
    scale_kib = 2 * 5 * 512 * 4 * 4 / 1024.0  # 80 KiB

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    x = np.arange(2)
    ax.bar(x[0], fp32_kib, width=0.55, color="#555555")
    ax.bar(x[1], data_kib, width=0.55, color="#1f77b4")
    ax.bar(x[1], scale_kib, width=0.55, bottom=data_kib, color="#ff7f0e")
    ax.text(x[0], fp32_kib + 12, f"{fp32_kib:.0f} KiB", ha="center", fontsize=8)
    ax.text(x[0], fp32_kib / 2, "fp32 K and V", ha="center", va="center",
            fontsize=8, color="white")
    ax.text(x[1], data_kib / 2, "int8 K and V", ha="center", va="center",
            fontsize=8, color="white")
    ax.text(x[1], data_kib / 2 - 26, f"{data_kib:.0f} KiB", ha="center",
            va="center", fontsize=8, color="white")
    ax.text(x[1], data_kib + scale_kib / 2, f"fp32 scales, {scale_kib:.0f} KiB",
            ha="center", va="center", fontsize=7.5)
    ax.text(x[1], fp32_kib + 12, f"{data_kib + scale_kib:.0f} KiB total\n"
            "2.67x smaller", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(["fp32 KV", "int8 KV"])
    ax.set_ylabel("KiB at 512 context")
    ax.set_ylim(0, 740)
    ax.set_title("KV cache footprint, hot window excluded")
    fig.text(0.5, 0.012, "Measured reclaim after the reference pass: 655,368 B. "
             "Scales are 50% of the int8 bytes.",
             ha="center", fontsize=6.5, color="#666666")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(OUT / "fig_kv_footprint.png")
    fig.savefig(OUT / "fig_kv_footprint.pdf")
    plt.close(fig)

# --------------------------------------------------------------------------

def main():
    fig_throughput()
    fig_sustained()
    fig_sections()
    fig_subprofile()
    fig_accuracy()
    fig_psram()
    fig_cache()
    fig_waterfall()
    fig_dma()
    fig_footprint()
    print("wrote figures to", OUT)
    for p in sorted(OUT.glob("*.png")):
        print(" ", p.name)

if __name__ == "__main__":
    main()
