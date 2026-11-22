# INT8 KV cache experiment: result

Date: 2026-11-21
Board: ESP32-S3-CAM on /dev/ttyACM0, 240 MHz, ESP-IDF v5.4.4
Model: dim 64, hidden 172, 5 layers, 8 heads, 4 KV heads, vocab 512, seq 512
Firmware: this tree, switches in `main/bench_cfg.h`

## What was changed

`KV_INT8 1` stores K and V in PSRAM as int8, one fp32 scale per head per
position. The 16-position hot window stays fp32 in internal SRAM. Positions
below `hot` are not quantised, because they are never read from the int8 cache.

Four attention variants were measured first, one build per change:

1. **Dequantise on read.** Each 8-value head is converted to fp32 and fed to
   the scalar dot.
2. **Integer K dot.** The query is quantised per head into a 16-byte vector
   with one zero half, and the score is one 16-lane `EE.VMULAS.S8.ACCX`
   instruction. ACCX is the explicit accumulator form; it has to be cleared
   before each dot because two dots cannot share it.
3. **V transpose.** V is stored as `[layer][kv_head][dim][pos]`, the V scale is
   folded into the softmax weight, the folded weights are quantised per head,
   and each output element is one SIMD dot over positions.
4. **Dual core.** The upper four heads run on a worker task pinned to core 1,
   with its own softmax and folded-weight buffers, one notification pair per
   layer.

Four more kernel and layout changes were then measured on top of the dual-core
build, again one build per change:

5. **Scale transpose.** K and V scales move from `[layer][pos][head]` to
   `[layer][head][pos]`, so one head walks contiguous floats instead of every
   fourth float.
6. **`softmax_fold`.** Normalisation, the V-scale fold, and the folded-weight
   max move into one pass over the scores.
7. **Batched K dot.** `llm_dot16x4_s8` computes four positions per call, so the
   query is loaded once and the call overhead is paid once per four positions.
8. **Inline rounding.** `lrintf` in the quantise loops is replaced by an inline
   `round_to_int`, removing a libcall per element.

The final build was then re-run three times to check the spread, and once more
at 32 KB cache. The accuracy harness is behind its own `KVCHECK` switch, so the
int8 weight path can be measured against fp32 KV without enabling the int8
cache.

Several changes were measured and rejected: DMA prefetch, a three-pass
`softmax_quant` fusion, tracking the score max inside the K loop, and
software-pipelining the four K loads. They are recorded below.

## Same-tree baseline matrix

The frozen tree's fp32 numbers predate the hot window, so the baseline was
re-measured in this tree, with the same allocator, the same hot window, and the
same harness. All four fp32 cells come from the fixed source that is described
under "Other negative results".

| Cache | Cores | Sustained tok/s | Attention us/token |
|---|---:|---:|---:|
| 32 KB | 1 | 48.85 | 19360.4 |
| 32 KB | 2 | 62.58 | 14867.9 |
| 64 KB | 1 | 58.06 | 16115.8 |
| 64 KB | 2 | 79.77 | 11287.7 |

Dual core is worth 1.281x at 32 KB and 1.374x at 64 KB. Doubling the cache is
worth 1.188x on one core and 1.275x on two, which is the first sign that the
scan is served by the cache and not by the PSRAM port alone.

At 64 KB the two effects are separable and they multiply almost exactly:

| Step | tok/s | Gain |
|---|---:|---:|
| fp32, single core | 58.06 | - |
| fp32, dual core | 79.77 | 1.374x |
| int8 KV + all kernels, dual core | 166.73 | 2.090x |
| **total** | **166.73** | **2.872x** |

The 2.090x row is the whole int8 bundle, storage plus kernels plus layout, not
the storage change alone. The waterfall below splits it. The product of the two
steps is 2.872, so the int8 kernel work did not lose anything to the second
core and the second core did not lose anything to the smaller cache.

## Optimization waterfall

Each row is one build with exactly one change, all at 64 KB cache. The chain
starts from the same-tree fp32 single-core baseline and ends at the final
build, so no step is inferred across trees.

| Build | Cores | Sustained tok/s | Step |
|---|---:|---:|---:|
| fp32 KV | 1 | 58.06 | - |
| int8 KV, dequantise on read | 1 | 46.52 | -19.9% |
| + integer K dot | 1 | 69.35 | +49.1% |
| + V transpose, folded scale | 1 | 89.82 | +29.5% |
| + dual core | 2 | 121.28 | +35.0% |
| + `att2` allocation fix | 2 | 123.22 | +1.6% |
| + scale transpose | 2 | 130.86 | +6.2% |
| + `softmax_fold` | 2 | 134.01 | +2.4% |
| + batched K dot x4 | 2 | 148.01 | +10.4% |
| + inline rounding | 2 | 166.73 | +12.6% |

The `att2` row is not an optimisation. It moves a buffer allocation out of the
`KV_INT8` guard and adds a pointer check, and the 1.6% step is the rebuild plus
the changed heap placement, not a code win. It is listed so the two sides of
the chain stay comparable.

The first four rows are the representation and mapping work; the last four are
the kernel and layout work. The naive int8 cache is a 19.9% regression on its
own, and it takes the integer K dot and the V transpose to get back above the
fp32 baseline. That is the central result: the byte saving is real, but it only
pays once the scan stops dequantising.

The kernel and layout steps, with the phase split:

| Build | tok/s | Step | Attention us | K dot us | softmax us | V quant us | V dot us |
|---|---:|---:|---:|---:|---:|---:|---:|
| post-fix baseline | 123.22 | - | 6665.9 | 5085.5 | 2913.8 | 3522.0 | 1310.0 |
| scale transpose | 130.86 | +6.2% | 6188.5 | 4596.9 | 2839.9 | 3130.3 | 1307.7 |
| `softmax_fold` | 134.01 | +2.4% | 6009.4 | 4652.6 | 3353.4 | 2200.4 | 1302.1 |
| batched K dot x4 | 148.01 | +10.4% | 5303.3 | 3301.2 | 3258.7 | 2236.0 | 1300.1 |
| inline rounding | 166.73 | +12.6% | 4608.1 | 3314.8 | 3305.3 | 874.5 | 1448.0 |

The four kernel and layout changes together are 1.353x on top of the dual-core
int8 baseline and cut attention wall time by 30.9%. The scale transpose, the
`softmax_fold`, the batched K dot, and the inline rounding are pure layout and
codegen wins: all four produce bit-identical logits. The batched K dot is
bit-identical because it computes the same four sixteen-lane MACs in the same
order.

The waterfall also shows where the time went. The K dot fell 35% (5086 to
3315 us) and the V quantise fell 75% (3522 to 875 us). The softmax phase grew
from 2914 to 3305 us even though the fold removed work from it, because the
phase boundaries changed: after `softmax_fold`, the phase that used to be
called "softmax" also contains the fold and the folded-weight max. The labels
are not comparable across the first and last rows; only the wall time is.

## Reproducibility

After the first run that established the build, the final build was confirmed
with twelve full-context runs across four capture sessions. Generate time per
512 tokens ranged from 3,070,535 to 3,070,927 us, which is 166.72 to 166.75
tok/s, a spread of 0.013%. The three runs inside one capture are tighter still.

The board was reflashed between sessions, so the spread covers flash, boot, and
allocator placement, not just run-to-run noise. The same firmware was also
rebuilt several times during the matrix work, and the number did not move,
which means the result does not depend on a lucky link-time layout.

## Section profile

The final forward, instrumented per section over the full-context runs
(`bench_kvint8_final3.log`, 1536 forwards, 166.73 tok/s):

| Section | us/token | share |
|---|---:|---:|
| attention | 4608.1 | 76.7% |
| ffn w1/w3 | 305.8 | 5.1% |
| qkv matmuls | 276.6 | 4.6% |
| w2 + residual | 264.1 | 4.4% |
| rope | 227.2 | 3.8% |
| silu | 142.0 | 2.4% |
| wo + residual | 98.9 | 1.6% |
| final + classifier | 82.7 | 1.4% |

Attention is 76.7% of the token, down from 82.1% before the kernel work. The
matmuls, rope, and the MLP are position independent and together are 1397 us,
or 23.3%. If the attention scan were free the token would fall to 1.40 ms, so
in principle there is 4.3x of headroom; in practice the scan already runs at
85% of the measured port, so the remaining lever is fewer bytes per position or
a cheaper score path, plus the 23.3% of serial work.

## Attention sub-profile

The attention section is split into four cycle-counted phases. Two different
clocks appear in this record, so they are defined here:

- **Wall time** is elapsed time per token from `esp_timer_get_time()`, the same
  clock that produces tok/s. `PROF attn` reports 4.608 ms of wall time.
- **Core time** is CPU cycles divided by 240 MHz, summed across both cores.
  The four APROF phases are core time. They can exceed wall time because the
  two cores run at the same time.

The four phases sum to 8,942.6 us of core time, which is 1.94x the 4.608 ms wall
time. The difference between 2 x 4.608 ms and 8,942.6 us is 2.97%, the per-layer
notification and join cost. The phases are not meant to add to 4.61 ms.

| Phase | us/token (core time, both cores) | share of attention core time |
|---|---:|---:|
| K dot (integer, SIMD) | 3314.8 | 37.1% |
| softmax + fold | 3305.3 | 37.0% |
| V dot (integer, SIMD) | 1448.0 | 16.2% |
| V weight quantise | 874.5 | 9.8% |

The K dot and the softmax are now tied. The K dot is at its instruction floor:
one 16-lane MAC per head per position, and the ACCX accumulator cannot overlap
two dots, so each position pays the clear, the MAC latency, and the read. The
softmax is a table-interpolated `exp` plus one normalise-and-fold pass, a few
tens of cycles per score, which is close to what the LUT form allows.

The V dot is slightly larger than in the intermediate builds (1300 to 1448 us).
The dot code itself is unchanged, so the difference is not a kernel effect: the
phase boundaries and the cache state differ between builds, and the V quantise
that feeds the dot was 4x cheaper in the final build.

## Cache sensitivity of the final kernel

The final build was run at both cache sizes. Everything else was held constant.

| Cache | tok/s | Attention us | K dot us | softmax us | V quant us | V dot us |
|---|---:|---:|---:|---:|---:|---:|
| 32 KB | 156.31 | 5117.9 | 3989.2 | 3398.6 | 875.0 | 1656.0 |
| 64 KB | 166.73 | 4608.1 | 3314.8 | 3305.3 | 874.5 | 1448.0 |

The 64 KB cache is 6.7% faster on sustained throughput. That is less than the
14.5% it was worth before the kernel work, and the reason is visible in the
phase split: the scale transpose made the scale reads contiguous, and the
batched K dot cut the number of K lines touched per position, so there is less
line reuse left for the cache to capture. The K dot is 17% faster and the V dot
13% faster with the larger cache; the softmax and the V quantise are essentially
flat because they work on internal buffers.

The logits are bit-identical at both cache sizes, so the difference is purely a
memory effect.

## PSRAM microbenchmark and scan bandwidth

The reference is a 512 KiB buffer in PSRAM, read sequentially as volatile
32-bit words, at 240 MHz with the 64 KB data cache enabled and octal PSRAM at
80 MHz. The reported figure is the best of three runs after one warm-up pass.
This is one access pattern under one configuration, not a physical maximum for
every pattern.

| Pattern | MB/s |
|---|---:|
| sequential read | 60.6 |
| strided read, 16 of every 32 B | 30.3 |
| sequential write | 36.8 |

At 512 context the int8 scan covers positions 16 to 511, because positions 0 to
15 stay in the fp32 hot window in internal SRAM. That is 5 layers x 496
positions x 96 B = 238,080 B of PSRAM traffic per token, where the 96 B is 32 B
of K, 32 B of V, 16 B of K scales, and 16 B of V scales. Divided by the 4.608 ms
attention wall time, that is 51.7 MB/s, or about 85% of the sequential read
bandwidth measured under the same test configuration.

The scan is therefore close to port limited. The remaining 15% covers the
strided K chunks, the scale reads that share cache lines with the data, and the
positions that are still cache resident from the previous token. Further
speedup has to come from moving fewer bytes, not from more aggressive prefetch,
which is why the DMA experiment lost.

## KV footprint and traffic

The 240 KiB is the combined KV payload and scale metadata, not 240 KiB of int8
data. For 5 layers, 4 KV heads, head dimension 8, and 512 positions:

| Component | Bytes | KiB |
|---|---:|---:|
| K payload, int8 | 81,920 | 80 |
| V payload, int8 | 81,920 | 80 |
| K scales, fp32 per head per position | 40,960 | 40 |
| V scales, fp32 per head per position | 40,960 | 40 |
| **combined allocation** | **245,760** | **240** |

The payload is 160 KiB (5 x 4 x 8 x 512 x 2 B) and the scales are 80 KiB
(5 x 4 x 512 x 2 x 4 B). The 16-position fp32 hot window is a separate
20,480 B in internal SRAM and is excluded from both this total and the PSRAM
traffic figure. At position 511 only 496 positions are written to and read from
the int8 cache, so the per-token traffic is 238,080 B (232.5 KiB) of the
245,760 B allocation.

The bench build keeps the fp32 K/V arrays so the reference forward can run on
the same model. After the last reference pass the arrays are freed and the
allocator reports the reclaim.

| Item | Bytes | KiB |
|---|---:|---:|
| fp32 K+V at 512 context | 655,360 | 640 |
| int8 K+V payload | 163,840 | 160 |
| fp32 scales, K and V | 81,920 | 80 |
| **int8 total** | **245,760** | **240** |
| measured reclaim | 655,368 | 640.0 |

The cache is 2.67x smaller, not 4x. The per-head fp32 scale costs 32 B per
position per layer, half the 64 B of int8 data. Halving the scale to bf16 would
take the total to 204,800 B (200 KiB, 3.2x). The 16-position hot window adds
20,480 B in both builds and is excluded from the table.

The reclaim is 8 bytes above the array size, which is allocator bookkeeping.
Free PSRAM after the drop is 6,911,464 B.

## Dual core

The worker task is created on the first forward pass, pinned to core 1 with a
4096-byte stack. `app_main` is already pinned to core 0 by Kconfig. Heads 0-3
run on the calling core, heads 4-7 on the worker, and the two halves use
separate `att` and `a8` buffers. One `xTaskNotifyGive` pair per layer keeps the
cores in step.

On the same-tree fp32 baseline the effect is 1.374x at 64 KB (58.06 to 79.77
tok/s) and 1.281x at 32 KB (48.85 to 62.58). Attention alone goes from 16.12 to
11.29 ms/token at 64 KB, a 1.428x speedup, not 2x. The gap is expected: the K/V
write and the query quantisation run before the split, both cores share the
PSRAM cache and the GDMA/PSRAM port, and each layer pays one notification pair.

The worker costs 4096 bytes of stack plus 2560 bytes of score and quantise
buffers. In the final build, free internal SRAM is 17679 B at the start of the
benchmark and 13231 B after the KV allocation, with a largest free block of
7680 B, enough for the largest allocation.

The `att2` buffer used to be allocated only under `KV_INT8`, so an fp32 build
with dual core would pass a null attention buffer to the worker. The allocation
was moved outside the guard and the pointer is checked. The same-tree fp32 dual
core runs above are what caught it.

## Measured cost model

The controlled matrix gives exchange rates for this workload:

| Change | Ratio | Measured on |
|---|---:|---|
| 32 KB to 64 KB cache, 1 core | 1.188x | fp32 |
| 32 KB to 64 KB cache, 2 cores | 1.275x | fp32 |
| 1 to 2 cores, 32 KB | 1.281x | fp32 |
| 1 to 2 cores, 64 KB | 1.374x | fp32 |
| int8 KV + kernels, vs fp32 2 cores, 64 KB | 2.090x | int8 |

Scope: these are measured performance and memory-traffic ratios for the
ESP32-S3 at 240 MHz with octal PSRAM at 80 MHz, this 260K-parameter model, and
512 context. They are not a predictor of KV-cache behaviour on other MCU-class
parts. A part with a wider PSRAM port, a larger cache, or a vector unit wider
than 128 bits would shift the balance between the cache, the port, and the
accumulator. Generalising the model would need the same matrix on additional
hardware and workloads.

## DMA prefetch: negative result

V was re-laid out in position tiles, `[layer][kv_head][tile][dim][tile_pos]`,
so a whole tile for one kv head is contiguous and one `esp_async_memcpy` call
can move all 8 dimension rows into internal SRAM. Tiles were double buffered so
the copy for tile t+1 overlaps the dot for tile t.

| V layout | DMA | Tile | Sustained tok/s | pos 384-511 tok/s | Attention ms/token |
|---|---|---:|---:|---:|---:|
| row `[dim][pos]` | no | - | 89.82 | 55.56 | 9.68 |
| tiled | no | 64 | 86.29 | 53.25 | 10.13 |
| tiled | yes | 64 | 49.56 | 30.54 | 18.76 |
| tiled | yes | 128 | 59.80 | 37.99 | 15.30 |
| tiled | yes | 256 | 61.97 | 41.72 | 14.70 |

These rows were measured before the scale transpose and the batched K dot, so
they compare layouts within one build, not against the final 166.73 tok/s.

The tiled layout alone costs 4%, which is the expected loss from losing the
long contiguous `[dim][pos]` rows. DMA then costs another 1.45x to 1.85x on
attention. Larger tiles recover part of it but plateau well short of the
no-DMA build.

The cause is the async memcpy path itself, not the transfer. Each tile pays a
source cache writeback (`esp_cache_msync`), a descriptor allocation, and an ISR
callback. At 512 to 2048 bytes per tile that fixed cost exceeds the benefit of
SRAM residency. The hardware cache already streams the long V rows efficiently,
so explicit DMA adds work without removing the bottleneck.

Accuracy is unchanged by the tiled layout or DMA: the 12-step int8 dumps are
bit-identical to the row layout, so the layout change is numerically neutral.
That 12-step comparison is not used for the accuracy claim, which is measured
at nine depths.

## Other negative results

**Three-pass `softmax_quant`.** Folding the quantisation into the `exp` pass,
so the weights are written once and the separate quantise loop disappears,
regressed from 134.01 to 131.95 tok/s. The extra compare and multiply inside
the `exp` loop cost more than the saved pass. The four-pass `softmax_fold` was
restored.

**Max in the K loop.** The score max was tracked inside the K dot loop, where
the core is stalled on ACCX, and the max scan was removed from `softmax_fold`.
This regressed from 166.73 to 162.60 tok/s (3149 us/token against 3071 us). The
K loop is not as latency-bound as the ACCX chain suggests, and the removed scan
was cheaper than the added per-position compare and select. Reverted.

**Pipelined K loads.** `llm_dot16x4_s8` was rewritten to issue all four K loads
before the first MAC, so the load latency overlaps the accumulator chain. It
made no difference: 166.23 tok/s against 166.73, with the K dot at 3351 us
against 3315. The ACCX dependency chain, not the load, is the floor. Reverted
to the simpler kernel.

**Bug found by the fp32 matrix.** `q8` selects the int8 weight path, not the
int8 KV cache, so it can be true in a `KV_INT8 0` build. The fold condition
used `q8`, which meant the fp32 build entered the fold loop with a null scale
pointer and panicked on the first attention call. The fp32 baselines that
predate `softmax_fold` never hit it, so it only appeared when the matrix was
re-run on the fixed source. The fold is now compiled out when `KV_INT8 0`. The
fix leaves the `KV_INT8 1` expansion unchanged, so the int8 results stand.

## Accuracy

The 12-step dump never leaves the hot window, so a separate 124-token paragraph
is repeated to fill all 512 positions. Logits are dumped at nine depths: the
hot-window boundary (15, 16, 17), the first int8 positions (32, 64), and the
deep regime (128, 256, 384, 511).

Three forward paths are compared at the same depths and on the same token
sequence:

1. **fp32 reference.** fp32 weights and fp32 KV.
2. **Weight-only.** int8 weights and fp32 KV. This is the `KV_INT8 0` build
   with the `KVCHECK` harness enabled.
3. **Combined.** int8 weights and int8 KV, the final build.

The fp32 reference is bit-identical in the two builds, and the token sequence
is the same at every depth.

| Position | Weight-only RMSE | KV-only RMSE | Combined RMSE | Combined cosine |
|---:|---:|---:|---:|---:|
| 15 | 2.128% | 0.000% | 2.128% | 0.9998874 |
| 16 | 1.125% | 1.601% | 2.089% | 0.9998994 |
| 17 | 2.086% | 2.870% | 1.960% | 0.9998322 |
| 32 | 1.794% | 3.092% | 2.905% | 0.9995908 |
| 64 | 1.426% | 5.622% | 4.564% | 0.9996197 |
| 128 | 1.165% | 1.827% | 1.543% | 0.9999312 |
| 256 | 1.724% | 3.453% | 3.235% | 0.9995052 |
| 384 | 0.881% | 2.657% | 2.712% | 0.9998220 |
| 511 | 0.789% | 1.362% | 1.077% | 0.9999461 |

Top-1 agrees at all nine depths on all three paths. The weight-only error is
0.789% to 2.128%, the KV-only error is 0% to 5.622%, and the combined error is
1.077% to 4.564%. The KV-only error is exactly zero at position 15, which is a
built-in control: position 15 is inside the fp32 hot window and is never
quantised, so the two int8 builds read the same bytes there.

The errors do not add linearly. At position 17 the combined error is smaller
than both components, and at 511 it is smaller than the KV-only component,
because the weight error and the KV error are independent vectors that can
partially cancel when projected onto the top logits. A single depth can
therefore look better than a shallower one. What matters is that there is no
systematic drift out to 511 positions, where 496 of the 512 attended positions
are int8.

The frozen tree reported 0.15% relative RMSE for int8 weights. That figure is
not reproducible from the frozen tree's own dumped logits, which give about
3.36% at the worst of the 12 steps, and the test stayed at position 16 or
below. The nine-depth harness here runs to 512 positions and measures 0.79% to
2.13% for the same weight path. The depth and the prompt differ, so the two are
not comparable, and this tree re-measures accuracy instead of quoting the
earlier number.

Two separate claims are being made, and they should be read separately.

**Optimisation preservation.** The int8 logits are bit-identical across the
scale transpose, `softmax_fold`, the batched K dot, the inline rounding, and the
32 KB cache build. All six builds produce the same nine depth dumps, so those
five steps are pure performance changes. The V transpose and the folded V scale
are part of the representation change, not a pure optimisation, and the dual
core is a mapping change; their effect is the accuracy comparison below.

**Accuracy against the fp32 reference.** The combined path agrees on top-1 at
all nine depths, with relative logit RMSE from 1.077% to 4.564% and cosine
similarity of at least 0.9995052. The KV cache contributes 0% to 5.622% on its
own; the weight path contributes 0.789% to 2.128%.

## Conclusion

The naive int8 KV was not a win. Against the same-tree fp32 baseline it was
19.9% slower at 64 KB (46.52 against 58.06 tok/s), because dequantising K to
fp32 cost more scalar work than the bytes saved. Seven changes turned it around:

1. Moving the K dot into the PIE unit: one 16-lane `EE.VMULAS.S8.ACCX` MAC
   replaces 8 conversions, 8 multiplies, and 8 adds per head per position.
2. Transposing V and folding the scale into the softmax weight: one SIMD dot
   per output element replaces 8 scalar multiply-adds per position.
3. Splitting the heads across both cores: 1.374x on the fp32 baseline.
4. Transposing the scales so each head reads contiguous floats.
5. Folding normalisation, the V scale, and the weight max into one pass.
6. Batching four K positions per call.
7. Replacing `lrintf` with inline rounding in the quantise loops.

The result was then checked at a second cache size and a second core count.

The result is 166.73 tok/s at 512 context, 2.872x over the same-tree fp32
single-core baseline and 2.090x over the same-tree fp32 dual-core baseline.
Attention is 4.61 ms of a 6.00 ms token and streams 238,080 B per token at
51.7 MB/s, about 85% of the sequential PSRAM read bandwidth measured under the
same test configuration.

The remaining attention cost is split almost evenly between the K dot and the
softmax, and both are close to their instruction floors on this core. The
serial part of each layer (qkv, rope, the MLP, the classifier) is 23.3% of the
token and the scan is the other 76.7%. A further gain therefore needs fewer
bytes per position or a cheaper score path, for example bf16 scales and a wider
batched dot, plus work on the 23.3% of serial sections. DMA does not help
because the scan is already served well by the cache and the async memcpy
overhead is larger than the latency it hides.

## Files

- `logs/bench_fp32_32_single.log`: same-tree fp32 baseline, 32 KB, one core
- `logs/bench_fp32_32_dual.log`: same-tree fp32 baseline, 32 KB, two cores
- `logs/bench_fp32_64_single.log`: same-tree fp32 baseline, 64 KB, one core
- `logs/bench_fp32_64_dual.log`: same-tree fp32 baseline, 64 KB, two cores
- `logs/bench_acc_ref.log`: `KV_INT8 0` with `KVCHECK 1`, int8 weights against
  fp32 KV at nine depths, the weight-only accuracy reference
- `logs/bench_kvint8_fix.log`: int8 dual core after the `att2` fix, start of
  the waterfall
- `logs/bench_kvint8_strans.log`: scale transpose
- `logs/bench_kvint8_fuse.log`: `softmax_fold`
- `logs/bench_kvint8_fuse2.log`: rejected three-pass `softmax_quant`
- `logs/bench_kvint8_x4.log`: batched K dot
- `logs/bench_kvint8_round.log`: inline rounding, first run of the final build
- `logs/bench_kvint8_final3.log`: final build, confirmation run, source of the
  section and phase profiles
- `logs/bench_kvint8_final.log`, `logs/bench_kvint8_final2.log`: the same final
  build in other sessions
- `logs/bench_kvint8_final_32kb.log`: final build at 32 KB cache
- `logs/bench_kvint8_maxfuse.log`: rejected max-in-K-loop
- `logs/bench_kvint8_x4p.log`: rejected pipelined K loads
- `logs/bench_kvfp32.log`: frozen-tree fp32 baseline, 32 KB, one core
- `logs/bench_kvint8.log`: dequant on read, 32 KB cache
- `logs/bench_kvint8_c64.log`: dequant on read, 64 KB cache, before hot skip
- `logs/bench_kvint8_c64_skip.log`: dequant on read, 64 KB cache, final
- `logs/bench_kvint8_idot.log`: integer K dot, before hot skip
- `logs/bench_kvint8_idot2.log`: integer K dot, final, with the 124-token
  accuracy check
- `logs/bench_kvint8_vt.log`: integer K dot + V transpose
- `logs/bench_kvint8_prof.log`: V transpose with the section profile
- `logs/bench_kvint8_dc.log`: first dual-core run
- `logs/bench_kvint8_tiled.log`: tiled V layout, no DMA
- `logs/bench_kvint8_dma.log`: tiled V layout + DMA, tile 64
- `logs/bench_kvint8_dma128.log`: tiled V layout + DMA, tile 128
- `logs/bench_kvint8_dma256.log`: tiled V layout + DMA, tile 256
- `logs/bench_kvint8_meas32.log`: instrumented run, 32 KB cache, pre-optimization
- `logs/bench_kvint8_meas64.log`: instrumented run, 64 KB cache, before the
  PSRAM probe was moved out of the heap
- `logs/bench_kvint8_meas64b.log`: instrumented run, 64 KB cache,
  pre-optimization, source of the old section and cache numbers
- `logs/bench_kvint8_mem.log`: same build, frees the fp32 caches after the
  reference pass and reports the footprint
- `logs/figs/`: generated figures (PNG and PDF), rebuilt by
  `python3 tools/make_figures.py`
- `sdkconfig.32kb.bak`: 32 KB cache config
- `sdkconfig.64kb.bak`: 64 KB cache config (best)
