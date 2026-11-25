# Vendor comparison and the fused-load A/B

Sources: two captures of each microbenchmark build, two captures of the
v1.1.0 firmware, six captures of the v1.2.0 firmware. Regenerate with
`python3 tools/micro_report.py`. The script fails if a capture is missing, if
two captures of the same kernel disagree by more than 0.5%, or if the v1.2.0
logits differ from the v1.1.0 logits.

## Setup

ESP32-S3 at 240 MHz, ESP-IDF v5.4.4, ESP-DSP v1.8.2 with
`CONFIG_DSP_OPTIMIZED=y`. Microbenchmark buffers are in internal SRAM, working
set 15,872 B, so the numbers are instruction cost rather than PSRAM traffic.
Every case is timed five times and the median is reported; the harness loop and
call cost 5.00 cycles, measured with an empty function.

The two kernels differ only in the long dot. `two-load` is the v1.1.0 loop:
two vector loads and a separate MAC per 16-byte block. `fused` is the v1.2.0
loop: `EE.VMULAS.S8.ACCX.LD.IP` loads the next block while accumulating the
current one, in a zero-overhead `loopnez`. The 16-lane single dot and the
four-position batch are unchanged between the two builds.

## Microbenchmark

| Kernel | Case | Blocks | Cycles/call | Cycles/block |
|---|---|---:|---:|---:|
| two-load | one 16-lane dot | 1 | 24.01 | 24.01 |
| two-load | ESP-DSP len 16 | 1 | 36.02 | 36.02 |
| two-load | four positions, one call | 4 | 64.04 | 16.01 |
| two-load | four positions, four calls | 4 | 89.06 | 22.26 |
| two-load | ESP-DSP four calls | 4 | 132.09 | 33.02 |
| two-load | len 32 | 2 | 42.03 | 21.01 |
| two-load | len 128 | 8 | 91.06 | 11.38 |
| two-load | len 496 | 31 | 276.17 | 8.91 |
| fused | len 32 | 2 | 38.03 | 19.01 |
| fused | len 128 | 8 | 57.03 | 7.13 |
| fused | len 496 | 31 | 127.08 | 4.10 |
| ESP-DSP | len 32 | 2 | 39.03 | 19.51 |
| ESP-DSP | len 128 | 8 | 56.03 | 7.00 |
| ESP-DSP | len 496 | 31 | 128.08 | 4.13 |

The four-position batch is worth 2.063x against four ESP-DSP calls and
1.39x against four calls to our own single dot. Replacing the batch
with the vendor primitive would add 17 cycles per score, about
1406 $\mu$s per token across 19,840 scores.

The vendor long dot is 2.16x faster per block than the
two-load kernel (4.13 against 8.91 cycles), which is why
the fused loop was adopted. The fused v1.2.0 kernel matches the vendor loop
(4.10 against 4.13 cycles per block) without the vendor
prologue, so it also wins at short lengths. On this workload the final kernel
is equivalent to ESP-DSP, not slower than it.

Correctness: each build prints 22 `CHECK` assertions. One check is one `CHECK`
line; the aggregate `MICRO correctness PASS` line summarises those 22 and is
not a 23rd check.

## End-to-end A/B

Protocol: alternate baseline and candidate in one session, 512 context, 64 KB
cache, dual core, three 512-token runs per capture. Baseline is v1.1.0
(2 captures, 6 runs); candidate is v1.2.0
(6 captures, 18 runs). Both use the same
`bench_cfg.h`, prompt and harness.

| Metric | v1.1.0 | v1.2.0 | Change |
|---|---:|---:|---:|
| Sampled throughput (tokens/s) | 166.74 | 168.25 | +0.91% |
| Decode-forward rate (forwards/s) | 165.11 | 166.61 | +0.91% |
| Sampled-token time (ms) | 5.998 | 5.944 | -0.90% |
| Attention (ms) | 4.608 | 4.554 | -1.18% |
| K dot ($\mu$s) | 3314.7 | 3297.0 | -0.54% |
| Softmax ($\mu$s) | 3305.3 | 3302.9 | -0.07% |
| V quantise ($\mu$s) | 874.5 | 874.4 | -0.02% |
| V dot ($\mu$s) | 1448.4 | 1355.7 | -6.40% |

All 42 logit blocks are byte-identical between the two builds, so the
kernel change is bit-exact. The spread over the 18 v1.2.0 runs
is 168.24 to 168.28 sampled tokens/s
(166.60 to 166.64
decode forwards/s), 0.023%.

## Conclusion

The vendor long dot was 2.16x faster per block than the
two-load kernel. Adopting its fused load closed that gap and moved the V phase
from 1448.4 to 1355.7 $\mu$s of core time
(-6.4%), worth
+0.91% end to end. The rest of the scan
did not move, which is what the bandwidth analysis predicts: the V dot was the
one phase with enough instruction headroom left to matter.
