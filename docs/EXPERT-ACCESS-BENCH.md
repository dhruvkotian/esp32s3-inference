# PSRAM-resident expert access benchmark

Status: **v1.4.0 engineering investigation. Not tagged, not evaluated, and not
part of any published result.** The firmware path is compiled but disabled by
default (`EXPERT_ACCESS_BENCH 0` in `firmware/main/bench_cfg.h`).

This is Experiment 0 plus the A/B/C memory-access benchmark for a possible
memory-resident sparse MoE. It answers one question before any format, router or
training work is justified:

> What does it cost to execute an int8 expert directly out of PSRAM on this
> board, compared with copying it into internal SRAM first?

Nothing else changes. `moe.h` limits, the `MOE1` format, `moe_hook()` and the
published v1.2.0/v1.3.x behaviour are untouched.

## Why it is standalone

The benchmark builds its own synthetic expert blocks. It does not load a model,
a sidecar or the SD card, so the measurement cannot be confused by the backbone,
the KV cache or the model manager. It exits from `bench_main()` before the mount.

The arithmetic is identical in every mode: all three call the same
`llm_mm_s8_generic()` wrapper, which is the per-row loop `mm_q_pre()` already
uses for any stride other than 64. The only variable is where the weight bytes
come from.

## Geometry

The shapes are the ones a 5.3M-parameter MoE with 8 experts and top-2 would use,
not the 260K model's shapes. This matters: `llm_mm_s8_64()` is selected only when
the weight stride is 64, so it is not the kernel that would run here.

| Tensor | Rows x cols | Row stride | int8 bytes |
|---|---:|---:|---:|
| `w1` | 264 x 96 | 96 | 25,344 |
| `w3` | 264 x 96 | 96 | 25,344 |
| `w2` | 96 x 264 | 272 | 26,112 |
| row scales | 624 rows | - | 2,496 |

`w2` is padded from 264 to 272 so the row stride stays a multiple of 16, which
the 128-bit loads require. Per layer that is **79,296 B**; per expert, over 8
layers, **634,368 B**. Those are the numbers the benchmark prints in `EA_GEOM`,
so the log is self-checking against this table.

## Modes

| Mode | Path | What it represents |
|---|---|---|
| **A** | PSRAM -> 79,296 B SRAM slice -> compute | The v1.3.0 `moe_hook()` refill, which copies a whole per-layer expert slice |
| **B** | PSRAM pointer -> compute | Zero-copy: the kernels read the mapping directly |
| **C** | PSRAM -> 1/2/4/8/16 KB tile -> compute | Bounded staging with two alternating SRAM buffers |
| **C-prefetch** | as C, plus a software touch of the next tile | Tests whether pulling lines in early helps |

All modes move 8 x 79,296 B out of PSRAM per expert forward, so the `MB/s`
column is directly comparable. Mode A splits `copy_us` from `comp_us`; B reports
compute only; C splits copy from compute per tile.

Two routing patterns are measured. `pinned` reuses one expert block; `alternating`
switches block per layer, which is the pattern that defeats any residual cache
reuse.

## Experiment 0: memory report

The normal boot path now logs one `MOE_MEM` line per stage:

| Stage | Where |
|---|---|
| `boot` | after `banner()` / `dot_selftest()` / `hw_bench()` |
| `pre_boot` | after the SD mount, before the model load |
| `backbone` | after `boot_model()` returns |
| `arena` | after `llm_print_mem()`, i.e. the int8 arena and KV cache are live |
| `moe_bound` | after `moe_bind()` in `MOE_TEST` |
| `warmup` | after the first int8 generation |

Each line carries `psram_free`, `psram_largest`, `internal_free` and
`internal_largest`. `psram_largest` is the number that decides whether an
expert sidecar can be one contiguous allocation; free bytes alone are not
enough.

To compare the model cache on and off, rebuild with `-DMODEL_CACHE_SLOTS=0`.
The cache loops become no-ops and peak PSRAM drops from four blobs to one.

## Running it

```bash
# 1. enable the benchmark
sed -i 's/#define EXPERT_ACCESS_BENCH 0/#define EXPERT_ACCESS_BENCH 1/' \
    firmware/main/bench_cfg.h

# 2. build and flash the research configuration
cd firmware
idf.py build
idf.py -p /dev/ttyACM0 flash monitor | tee ../logs/expert_access.log

# 3. summarise
cd ..
python3 tools/analyze_expert_access.py logs/expert_access.log --csv /tmp/ea.csv

# 4. restore the published default
sed -i 's/#define EXPERT_ACCESS_BENCH 1/#define EXPERT_ACCESS_BENCH 0/' \
    firmware/main/bench_cfg.h
```

The run takes well under a second of compute; almost all of the wall time is the
UART output. `EA_EQUIV` must report `a_vs_b=1 a_vs_c=1`, otherwise the three
paths are not doing the same arithmetic and the timings are not comparable.

## What to record

- `EA_MEM` on the real board, with the published model booted, both with the
  model cache on and off.
- The full `EA_RUN` table for both patterns.
- `EA_EQUIV` and `EA_CTRL64` for the same capture.

## How to read the result

Mode A and mode B move the same number of PSRAM bytes. Any gap between them is
therefore the cost or benefit of the SRAM round trip against cached 128-bit
loads from the mapping:

- **B clearly faster than A**: the staging copy is pure overhead and the
  zero-copy view is the right design. Move to the 4-expert direct-PSRAM
  prototype.
- **B clearly slower than A**: the cached loads are stalling on line fills and
  the copy path is doing useful work by turning many small accesses into one
  streaming copy. Tiled streaming (C) becomes the candidate, and the 128-bit
  load from PSRAM is the thing to fix.
- **A and B within noise**: the design is bandwidth-bound either way, and the
  decision should be made on SRAM headroom rather than throughput.

Mode C only wins if bounded tiles keep the pipeline busier than one large copy.
On a single core, copy and compute cannot overlap without DMA, so expect C to
lose slightly to A. That is a useful result in itself: it means the tile buffers
are not worth their SRAM cost until GDMA is in the picture.

`EA_CTRL64` is a control, not the headline. It shows what the stride-64
specialised kernel buys over the generic one on a geometry that actually selects
it, which is a different model from the one under test.

## Cache and prefetch caveat

The 64 KB D-cache has 32-byte lines, so one line supplies two aligned 128-bit
loads. Do not assume every vector load misses; assume the *line* is fetched once
and never reused, because each weight byte is touched exactly once per forward.
The benchmark measures the consequence rather than asserting it.

The configured build exposes no explicit D-cache preload entry point, so the
`C-prefetch` variant does the only thing available in software: read one byte
per line to pull the tile in early. Its touch time is included in `copy_us`.
Compare it with plain C before concluding anything about prefetch.

## Predicted, not measured

From the shapes alone, if PSRAM sustains about 55 MB/s then one expert forward
is 634,368 B / 55 MB/s = **11.5 ms**, and the 76,032 MACs per layer are about
**0.6 ms** for all 8 layers. Copy cost would dominate compute by roughly 19x.
These are predictions to check, not results; the point of the benchmark is to
replace them with measurements.
