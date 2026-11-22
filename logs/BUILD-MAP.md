# Artifact build map

Each row of Table IV in the paper is one build of this firmware, and each row
changes exactly one switch or one kernel. This file maps the table rows to the
raw capture that produced them, so a reader can check every number without
trusting the paper.

Sustained throughput is the median of the `RUN ctx` lines in each capture:
`512 / gen_us * 1e6`, where `gen_us` is the decode wall time for 512 tokens
after a five-token prompt. `tools/analyze_bench.py` and
`tools/make_figures.py` use the same definition.

| Table IV row | tokens/s | Raw capture |
|---|---:|---|
| fp32 KV, 64 KB, 1 core | 58.06 | `bench_fp32_64_single.log` |
| int8 KV, dequantise on read | 46.52 | `bench_kvint8_c64_skip.log` |
| + integer K dot | 69.35 | `bench_kvint8_idot2.log` |
| + V transpose, folded scale | 89.82 | `bench_kvint8_vt.log` |
| + dual core | 121.28 | `bench_kvint8_dc.log` |
| + `att2` allocation fix | 123.22 | `bench_kvint8_fix.log` |
| + scale transpose | 130.86 | `bench_kvint8_strans.log` |
| + `softmax_fold` | 134.01 | `bench_kvint8_fuse.log` |
| + batched K dot x4 | 148.01 | `bench_kvint8_x4.log` |
| + inline rounding | 166.73 | `bench_kvint8_final3.log` |

`bench_kvint8_mem.log` repeats the `att2` build with the memory and phase
reporting enabled, which is why it also lands at 123.22 tokens/s.

## Same-tree fp32 matrix (Table III)

| Cache | Cores | tokens/s | Attention us/token | Raw capture |
|---|---:|---:|---:|---|
| 32 KB | 1 | 48.85 | 19,360.4 | `bench_fp32_32_single.log` |
| 32 KB | 2 | 62.58 | 14,867.9 | `bench_fp32_32_dual.log` |
| 64 KB | 1 | 58.06 | 16,115.8 | `bench_fp32_64_single.log` |
| 64 KB | 2 | 79.77 | 11,287.7 | `bench_fp32_64_dual.log` |

## Cache sensitivity (Fig. 6)

| Build | tokens/s | Raw capture |
|---|---:|---|
| 32 KB final kernel | 156.31 | `bench_kvint8_final_32kb.log` |
| 64 KB final kernel | 166.73 | `bench_kvint8_final3.log` |

The 14.5% figure quoted for the pre-kernel comparison comes from
`bench_kvint8_meas32.log` (106.74 tokens/s) and `bench_kvint8_meas64b.log`
(122.27 tokens/s), which are the same kernel with the cache doubled.

## Negative results (Table VI and Section 10.2)

| Variant | tokens/s | Raw capture |
|---|---:|---|
| tiled V, no DMA | 86.29 | `bench_kvint8_tiled.log` |
| tiled V + DMA, 64 B tiles | 49.56 | `bench_kvint8_dma.log` |
| tiled V + DMA, 128 B tiles | 59.80 | `bench_kvint8_dma128.log` |
| tiled V + DMA, 256 B tiles | 61.97 | `bench_kvint8_dma256.log` |
| three-pass `softmax_quant` | 131.95 | `bench_kvint8_fuse2.log` |
| max tracked inside the K loop | 162.60 | `bench_kvint8_maxfuse.log` |
| four K loads issued before the first MAC | 166.23 | `bench_kvint8_x4p.log` |

## Reproducibility runs

`bench_kvint8_final.log`, `bench_kvint8_final2.log`, `bench_kvint8_final4.log`
and `bench_kvint8_round.log` are independent captures of the final build. With
`bench_kvint8_final3.log` they give the spread reported in Section 8.5.

## Accuracy

`bench_kvint8_final3.log` carries the nine-depth logit dumps: `LOGD` blocks
labelled `fp32`, `int8`, `fp32L*` and `int8L*`, terminated by `LOGE`. The
`*L15` label is the control inside the fp32 hot window, where the KV-only error
is exactly zero. `data/expected.json` holds the reference logits used by the
accuracy harness.

## Notes

- `bench_kvfp32.log` and `bench_kvint8.log` are the earliest builds in this
  tree, before the hot window and the allocator were settled. They are kept
  because Section 7 refers to the 46.03 tokens/s starting point, but they are
  not rows of Table IV.
- `bench_acc_ref.log` repeats the fp32 dual-core cell with the accuracy harness
  compiled in, to confirm the harness does not perturb the baseline.
