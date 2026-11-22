# Vendor versus same-tree int8 dot kernels

Two captures of the same microbenchmark, `micro_dsp_vs_ours.log` and
`micro_dsp_vs_ours_run2.log`, on the paper's board: ESP32-S3 at 240 MHz,
ESP-IDF v5.4.4, ESP-DSP v1.8.2 (`CONFIG_DSP_OPTIMIZED=y`), 16 KB working set
in internal SRAM. The two captures agree to within 0.002% on every
case. Regenerate this file with `python3 tools/micro_report.py`.

The benchmark answers one question: on the shapes the attention scan actually
uses, what does the vendor primitive cost next to ours? The data is resident,
so these are instruction costs, not PSRAM traffic.

| Case | Blocks | Cycles/call | Cycles/block | Cycles/block less loop |
|---|---:|---:|---:|---:|
| harness loop and call, no kernel | 0 | 5.00 | 0.00 | 0.00 |
| ours, one 16-lane dot | 1 | 24.01 | 24.01 | 19.01 |
| ESP-DSP, len 16 | 1 | 36.02 | 36.02 | 31.02 |
| ours, len 32 | 2 | 42.03 | 21.01 | 18.51 |
| ESP-DSP, len 32 | 2 | 39.03 | 19.51 | 17.01 |
| ours, len 128 | 8 | 91.06 | 11.38 | 10.76 |
| ESP-DSP, len 128 | 8 | 56.03 | 7.00 | 6.38 |
| ours, len 496 | 31 | 276.17 | 8.91 | 8.75 |
| ESP-DSP, len 496 | 31 | 128.08 | 4.13 | 3.97 |
| ours, four positions, one call | 4 | 64.04 | 16.01 | 14.76 |
| ours, four positions, four calls | 4 | 89.06 | 22.26 | 21.01 |
| ESP-DSP, four positions, four calls | 4 | 132.09 | 33.02 | 31.77 |

The harness loop and call cost 5.00 cycles, measured by timing an
empty function. Subtract that when comparing single 16-lane dots; it is spread
over the blocks for the longer cases.

## What the numbers say

**Our batched K dot wins, and it has to.** Four separate ESP-DSP calls cost
132.09 cycles; one `llm_dot16x4_s8` call costs
64.04, a factor of 2.06. Four separate calls
to our own single dot cost 89.06, so the batch is
worth 1.39x against our own
code. Replacing the batch with the vendor primitive would add
17.01 cycles
per score, or about 1406 us per token across 19,840 scores,
roughly 23% of the 6.00 ms token.

**The vendor long dot wins, but there is little to collect.** At len 496 the
vendor loop is 2.16x
faster per block (4.13 against
8.91 cycles), because it fuses the load into
`ee.vmulas.s8.accx.ld.ip` and runs a zero-overhead `loopnez`. Our kernel loads
and multiplies separately. The V path calls a long dot once per output element:
160 calls per token over an average of 248.5 positions, about
2,485 blocks. Closing the whole per-block gap would save at most
49 us per token, 0.8% of the token,
and only if that phase were compute-bound, which the in-situ profile says it is
not. At len 16 the vendor call is the slower one
(36.02 against 24.01 cycles),
because its prologue and loop setup dominate, and many V calls at shallow
positions are that short.

**The instruction was never the contribution.** The vendor loop and ours issue
the same 16-lane MAC; the difference is two extra instructions per block and a
loop branch. Batching four scores per call is worth more than the vendor's
per-block advantage, and the paper's 40 cycles per score in situ is mostly
memory, not arithmetic.

## Why this does not change the paper

The paper claims a mapping and a cache layout, not a faster instruction. The
microbenchmark supports that: the kernel we wrote is better than the vendor
primitive on the shape the score path uses, and behind it only on the long dot,
where the measured phase is memory-bound and the ceiling is about
0.8%. Adopting the fused load in `llm_dot_s8` is a
bounded, testable change for a later version, not a correction to this one.
