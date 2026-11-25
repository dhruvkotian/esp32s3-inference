# Artifact build map

Each row of Table 5 in the paper is one build of this firmware, and each row
changes exactly one switch or one kernel. This file maps the table rows to the
raw capture that produced them, so a reader can check every number without
trusting the paper.

The tabulated throughput is the median of the `RUN ctx` lines in each capture:
`512 / gen_us * 1e6`, where `gen_us` is the decode wall time for 512 samples
after a five-token prompt. This is the sampled-token rate.

The paper's primary throughput is the decode-forward rate, 507 divided by the
same wall time. The harness performs 507 decode forwards because the last five
sampled tokens reuse the final logits. The two rates differ by the constant
factor 507/512, so at the final build 168.25 sampled tokens/s is 166.6 decode
forwards/s. Every configuration in this file uses the same harness, so the
ratios are unchanged. `tools/analyze_bench.py` and `tools/make_figures.py` use
the sampled-token definition.

| Table 5 row | sampled tok/s | Raw capture |
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
| + fused long dot | 168.25 | `bench_kvint8_fused_1.log` |

`bench_kvint8_mem.log` repeats the `att2` build with the memory and phase
reporting enabled, which is why it also lands at 123.22 tokens/s.

## Same-tree fp32 matrix (Table 4)

| Cache | Cores | sampled tok/s | Attention us/token | Raw capture |
|---|---:|---:|---:|---|
| 32 KB | 1 | 48.85 | 19,360.4 | `bench_fp32_32_single.log` |
| 32 KB | 2 | 62.58 | 14,867.9 | `bench_fp32_32_dual.log` |
| 64 KB | 1 | 58.06 | 16,115.8 | `bench_fp32_64_single.log` |
| 64 KB | 2 | 79.77 | 11,287.7 | `bench_fp32_64_dual.log` |

## Cache sensitivity (Fig. 6)

| Build | sampled tok/s | Raw capture |
|---|---:|---|
| 32 KB final kernel | 157.48 | `bench_kvint8_fused_32kb.log` |
| 64 KB final kernel | 168.25 | `bench_kvint8_fused_1.log` |

The 14.5% figure quoted for the pre-kernel comparison comes from
`bench_kvint8_meas32.log` (106.74 tokens/s) and `bench_kvint8_meas64b.log`
(122.27 tokens/s), which are the same kernel with the cache doubled.

## Negative results (Table 7 and Section 10.2)

| Variant | sampled tok/s | Raw capture |
|---|---:|---|
| tiled V, no DMA | 86.29 | `bench_kvint8_tiled.log` |
| tiled V + DMA, 64 B tiles | 49.56 | `bench_kvint8_dma.log` |
| tiled V + DMA, 128 B tiles | 59.80 | `bench_kvint8_dma128.log` |
| tiled V + DMA, 256 B tiles | 61.97 | `bench_kvint8_dma256.log` |
| three-pass `softmax_quant` | 131.95 | `bench_kvint8_fuse2.log` |
| max tracked inside the K loop | 162.60 | `bench_kvint8_maxfuse.log` |
| four K loads issued before the first MAC | 166.23 | `bench_kvint8_x4p.log` |

## Reproducibility runs

`bench_kvint8_fused_1.log` through `bench_kvint8_fused_6.log` are independent
captures of the final build, with the board reflashed between captures. They
give the 168.24 to 168.28 sampled tokens/s spread (166.60 to 166.64 decode
forwards/s) reported in Section 8. The older
`bench_kvint8_final*.log` captures are the v1.1.0 kernel and remain the source
for the `inline rounding` row.

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

## Vendor comparison (Section 4.5, Table 3)

The microbenchmark is a separate app. `tools/micro_report.py` regenerates
`logs/MICRO-RESULT.md` from the four captures below and fails if two captures
of the same kernel disagree by more than 0.5%.

| Case | Raw capture |
|---|---|
| two-load kernel, first run | `micro_dsp_vs_ours.log` |
| two-load kernel, second run | `micro_dsp_vs_ours_run2.log` |
| fused kernel, first run | `micro_dsp_vs_ours_fused.log` |
| fused kernel, second run | `micro_dsp_vs_ours_fused_run2.log` |

The app is in `micro/`. It links ESP-DSP v1.8.2, which is not vendored here;
`micro/README.md` gives the clone command and the build switches. The
`dot_s8.S` in `micro/main/` is a byte-identical copy of
`firmware/main/dot_s8.S`, so the timed kernels are the released kernels.

## Fused-load A/B (Section 4.5)

Baseline is v1.1.0 (`bench_ab_base_1.log`, `bench_ab_base_2.log`); candidate is
v1.2.0 (`bench_kvint8_fused_1.log` through `bench_kvint8_fused_6.log`). The
captures were interleaved in one session, with the board reflashed between
each. `tools/micro_report.py` recomputes the medians and fails if the 42 logit
blocks are not byte-identical.

| Metric | v1.1.0 | v1.2.0 |
|---|---:|---:|
| Sampled throughput (tokens/s) | 166.74 | 168.25 |
| Decode-forward rate (forwards/s) | 165.1 | 166.6 |
| Token (ms) | 5.998 | 5.944 |
| Attention (ms) | 4.608 | 4.554 |
| K dot (us) | 3314.7 | 3297.0 |
| Softmax (us) | 3305.3 | 3302.9 |
| V quantise (us) | 874.5 | 874.4 |
| V dot (us) | 1448.4 | 1355.7 |
