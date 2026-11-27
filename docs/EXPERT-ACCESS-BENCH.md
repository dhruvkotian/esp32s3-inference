# PSRAM-resident expert access benchmark

Status: **v1.4.0 engineering investigation. Not tagged, not evaluated, and not
part of any published result.** The benchmark is compiled but disabled by
default (`EXPERT_ACCESS_BENCH 0` in `firmware/main/bench_cfg.h`).

This is an isolated synthetic FFN weight-access benchmark. It is **not** a MoE
forward pass: there are no trained experts, no router and no weighted
expert-output combination. It answers one question before any of that is
justified:

> What does it cost to execute an int8 expert directly out of PSRAM on this
> board, compared with copying it into internal SRAM first?

Nothing else changes. `moe.h` limits, the `MOE1` format, `moe_hook()` and the
published v1.2.0/v1.3.x behaviour are untouched.

## Two separate captures

The benchmark runs inside `bench_main()` and returns before the SD mount and the
model load. That is deliberate: it isolates expert memory access from the
backbone, the KV cache and the model manager. The consequence is that **the
benchmark cannot report runtime PSRAM availability**, because in that build the
runtime does not exist yet.

So there are two captures, not one:

| Capture | Configuration | What it measures |
|---|---|---|
| **1. Runtime memory** | `BENCH_MODE 0` | Real PSRAM availability after the backbone and KV cache are live |
| **2. A/B/C access** | `BENCH_MODE 1`, `EXPERT_ACCESS_BENCH 1` | Isolated expert weight access |

Capture 1 must be repeated twice: once with the three inactive model-cache slots
enabled, once with them disabled. Only the largest contiguous block answers the
question "can an expert sidecar be one allocation?"

## Geometry

The shapes are the ones a 5.3M-parameter MoE with 8 experts and top-2 would use,
not the 260K model's shapes. This matters: `llm_mm_s8_64()` is selected only when
the weight stride is 64, so it is not the kernel that would run here. The
benchmark uses `llm_mm_s8_generic()`, the same per-row loop `mm_q_pre()` already
uses for any other stride.

| Tensor | Rows x cols | Row stride | int8 bytes |
|---|---:|---:|---:|
| `w1` | 264 x 96 | 96 | 25,344 |
| `w3` | 264 x 96 | 96 | 25,344 |
| `w2` | 96 x 264 | 272 | 26,112 |
| row scales | 624 rows | - | 2,496 |

`w2` is padded from 264 to 272 so the row stride stays a multiple of 16, which
the 128-bit loads require. Per layer that is **79,296 B**; per expert, over 8
layers, **634,368 B**. `EA_GEOM` prints these, so the log self-checks against
this table.

## Modes

| Mode | Path | What it represents |
|---|---|---|
| **A** | PSRAM -> 79,296 B SRAM slice -> compute | The v1.3.0 `moe_hook()` refill |
| **B** | PSRAM pointer -> compute | Zero-copy: the kernels read the mapping directly |
| **C** | PSRAM -> 1/2/4/8/16 KB tile -> compute | Bounded staging with two alternating SRAM buffers |
| **C-prefetch** | as C, plus a software touch of the next tile | Tests whether pulling lines in early helps |

All modes move 8 x 79,296 B out of PSRAM per expert forward, so `MB/s` is
comparable. Mode A splits `copy_us` from `comp_us`; B reports compute only; C
splits copy from compute per tile.

Two routing patterns are measured:

- `pinned` reuses expert 0 for all eight layers.
- `alternating` switches between expert 0 and expert 1 per layer.

`alternating` is a **per-layer** pattern within one forward pass. It does not
model a router changing its choice between consecutive tokens at the same layer.
That is a separate test, deferred until this one is understood.

## Equivalence checks

Two levels, because the first alone is not sufficient:

- `EA_ACC` compares the aggregate float sum across all layers. Cheap sanity
  check; two different vectors can share a sum.
- `EA_EQUIV` compares **every element** of the W1, W3 and W2 outputs on layer 0,
  for A vs B and A vs C, and reports `max` absolute difference and the count of
  differing elements.

All modes run identical integer dots on identical bytes, so `ok=1`,
`max=0.000000` and `diff=0` are achievable and expected. If any differ, the
timings are not comparable and the run is void.

## Capture 1: runtime memory

Run each block from the repository root; the trailing `cd ..` returns there.

```bash
# normal firmware, model cache enabled (the published default)
sed -i 's/#define BENCH_MODE 1/#define BENCH_MODE 0/' firmware/main/bench_cfg.h
cd firmware && idf.py build
idf.py -p /dev/ttyACM0 flash monitor | tee ../logs/moe_mem_cache3.log

# same, with the inactive model cache disabled
sed -i 's/#define MODEL_CACHE_SLOTS 3/#define MODEL_CACHE_SLOTS 0/' \
    main/model_lib.h
idf.py build
idf.py -p /dev/ttyACM0 flash monitor | tee ../logs/moe_mem_cache0.log

# restore both defaults
sed -i 's/#define MODEL_CACHE_SLOTS 0/#define MODEL_CACHE_SLOTS 3/' \
    main/model_lib.h
sed -i 's/#define BENCH_MODE 0/#define BENCH_MODE 1/' main/bench_cfg.h
cd ..
```

Each run logs one `MOE_MEM` line per stage:

| Stage | Where |
|---|---|
| `boot` | after `banner()` / `dot_selftest()` / `hw_bench()` |
| `pre_boot` | after the SD mount, before the model load |
| `backbone` | after `boot_model()` returns |
| `arena` | after `llm_print_mem()`, i.e. int8 arena and KV cache live |
| `moe_bound` | after `moe_bind()` in `MOE_TEST` |
| `warmup` | after the first int8 generation |

Each line carries `psram_free`, `psram_largest`, `internal_free` and
`internal_largest`. `psram_largest` is the number that matters; free bytes alone
hide fragmentation.

## Capture 2: A/B/C access

```bash
sed -i 's/#define EXPERT_ACCESS_BENCH 0/#define EXPERT_ACCESS_BENCH 1/' \
    firmware/main/bench_cfg.h
cd firmware && idf.py build
idf.py -p /dev/ttyACM0 flash monitor | tee ../logs/expert_access.log

cd ..
python3 tools/analyze_expert_access.py logs/expert_access.log --csv /tmp/ea.csv
```

Keep the 64 KB D-cache configuration exactly as published; do not change the
cache size or PSRAM frequency for this run. Preserve the raw UART capture after
analysing it.

Restore the published default and confirm it still builds:

```bash
sed -i 's/#define EXPERT_ACCESS_BENCH 1/#define EXPERT_ACCESS_BENCH 0/' \
    firmware/main/bench_cfg.h
cd firmware && idf.py build
```

## Reading the output

The analyser separates **cold** (D-cache evicted before the run) from
**no-evict** (no eviction sweep) because pooling them hides the difference.
"no-evict" is not "warm": the 634 KB working set is far larger than the 64 KB
cache, so the expert is never fully cache-resident. The label only records that
no sweep was performed. The firmware emits the
state on every record (`cold=1` / `cold=0`), so the grouping stays correct even
when the eviction policy changes. Compare modes on `total_us`, not on
`comp_us`: for A that is copy plus compute, for B it is the direct compute
alone.

| Result | Engineering decision |
|---|---|
| B clearly faster than A | Develop direct PSRAM expert execution |
| B clearly slower than A | Investigate tiled staging and memory access |
| C beats A | Investigate an optimised bounded-buffer path |
| C loses to A | Don't spend SRAM on tiles without a demonstrated benefit |
| C-prefetch clearly beats C | Investigate cache preloading or DMA |
| All similar | Prefer the simpler design with lower SRAM use |

Mode B must beat Mode A on **total elapsed time**, not merely on compute time.
`T_A = T_copy + T_compute`, `T_B = T_direct_compute`.

### Sampling cold and no-evict

By default only run 0 is cold, so `EA_RUNS = 10` gives **one cold and nine
no-evict** observations. That is not ten independent cold runs, and it must not
be read as one. Two settings control this, both `#ifndef`-guarded:

| Setting | Effect |
|---|---|
| `EA_RUNS` (default 3) | Repeats per configuration |
| `EA_COLD_EVERY` (default 0) | `1` evicts the D-cache before every run |

For a stable **no-evict** comparison, raise only `EA_RUNS`. For a stable **cold**
comparison, set `EA_COLD_EVERY 1` and raise `EA_RUNS`; every run is then an
independent cold observation, and the analyser reports them all as `cold`:

```bash
sed -i 's/#define EA_RUNS       3/#define EA_RUNS       10/' \
    firmware/main/expert_access_bench.c
sed -i 's/#define EA_COLD_EVERY 0/#define EA_COLD_EVERY 1/' \
    firmware/main/expert_access_bench.c
```

Restore both to `3` and `0` afterwards. Do not mix the two modes in one capture;
they answer different questions.

## Caveats

- **Mode C is not a pure streaming comparison.** It stages weight tiles but
  still reads the row-scale vector directly from PSRAM, whereas A copies the
  scales along with the weights.
- **Prefetch is software.** `C-prefetch` performs an ordinary CPU pre-touch, one
  byte per 32-byte line. It does not use the hardware manual D-cache preload,
  and the configured ESP-IDF 5.4.4 build does not expose a public API for that.
  Do not call ROM cache functions without checking compatibility first.
- **Cache lines, not vector loads.** One 32-byte line supplies two aligned
  128-bit loads. Each weight byte is touched once per forward, so the line is
  fetched once and never reused. The benchmark measures the consequence rather
  than asserting it.
- **Single core, synchronous tiles.** Copy and compute cannot overlap without
  DMA, so C is expected to lose slightly to A. If it does, that does **not**
  disprove a genuinely pipelined DMA implementation; it only says synchronous
  tiling is not worth the SRAM on its own. Tiles become a candidate again only
  if prefetch helps materially or a DMA path is built and measured.

## Predicted, not measured

From the shapes alone, at about 55 MB/s one expert forward is
634,368 B / 55 MB/s = **11.5 ms**, and the 76,032 MACs per layer are about
**0.6 ms** for all 8 layers. Copy cost would dominate compute by roughly 19x.
These are predictions to check, not results.

## What to send back

| Output | What it tells us |
|---|---|
| `logs/expert_access.log` | Complete raw A/B/C capture |
| `/tmp/ea.csv` | Grouped cold/no-evict timings |
| `EA_ACC`, `EA_EQUIV` | Arithmetic consistency, per matrix |
| `EA_CTRL64` | Stride-64 kernel control |
| `EA_GEOM` | Actual tensor sizes used |
| `logs/moe_mem_cache3.log`, `logs/moe_mem_cache0.log` | Runtime PSRAM availability |
| Boot banner lines | Chip, flash and PSRAM configuration of the actual module |
