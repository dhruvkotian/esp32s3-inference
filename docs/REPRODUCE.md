# Reproducing the measurements

## Toolchain

| Item | Value |
|---|---|
| ESP-IDF | v5.4.4 |
| Target | esp32s3 |
| Toolchain | xtensa-esp32s3-elf, shipped with ESP-IDF v5.4.4 |
| Host tools | Python 3, pyserial, numpy, matplotlib |

The IDF version matters. `CONFIG_COMPILER_OPTIMIZATION_PERF=y` and the LX7
code generation change between IDF releases, and the paper's cycle counts are
tied to this one.

## Build

The paper's primary configuration is the 64 KB D-cache build. From a fresh
extraction, the tested sequence is:

```bash
. /path/to/esp-idf-v5.4.4/export.sh
cd firmware
idf.py set-target esp32s3
cp ../config/sdkconfig.64kb sdkconfig
idf.py fullclean
idf.py build
```

The default configuration builds the 32 KB D-cache variant. To reproduce the
paper's primary 64 KB configuration, replace the generated sdkconfig with
config/sdkconfig.64kb before building. The copy overwrites the sdkconfig that
`set-target` generated, and `fullclean` removes any previous build tree.

The 32 KB variant is the baseline used in the cache sweep, not the
configuration behind the headline result. To build it, omit the `cp` line:

```bash
. /path/to/esp-idf-v5.4.4/export.sh
cd firmware
idf.py set-target esp32s3
idf.py fullclean
idf.py build
```

With ESP-IDF v5.4.4, the 64 KB sequence reproduces the released image:
`.flash.text`, `.flash.rodata`, `.dram0.data` and `.iram0.text` are
byte-identical to the v1.2.0 build, and only the build timestamp and image
digest in the ESP-IDF app descriptor differ.

The CMake project name is `llm_s3_test`, inherited from the earlier tree, so
the binary is `build/llm_s3_test.bin`. The name is kept unchanged so that the
released source is the source that produced the paper.

`partitions.csv` still declares `model` and `tok` data partitions from the
earlier tree. This firmware reads both files from the microSD card and does not
use those partitions, but they are left in place because the flashed image
includes them.

## Model files

Copy `model/stories260K.bin` and `model/tok512.bin` to the root of a FAT
microSD card:

```
/sdcard/stories260K.bin
/sdcard/tok512.bin
```

The firmware opens those paths at boot. A missing file stops the run before
any timing starts.

## Flash

Application only, at `0x10000`:

```bash
esptool.py --port /dev/ttyACM0 --baud 921600 --chip esp32s3 write_flash \
  --flash_mode dio --flash_freq 80m --flash_size 16MB \
  0x10000 build/llm_s3_test.bin
```

Do not erase the whole flash unless you are prepared to rewrite the bootloader
and partition table as well.

## Capture

The CH343 bridge has no auto-reset wiring that matches esptool's inverted
DTR/RTS sequence, so `capture_until.py` pulses EN explicitly:

```bash
python3 tools/capture_until.py /dev/ttyACM0 240 logs/bench_mine.log BENCH_DONE
```

The run ends at the `BENCH_DONE` marker. Every log in `logs/` was captured this
way.

## Build switches

`firmware/main/bench_cfg.h` is the experiment control. The final build uses:

| Switch | Final value | Meaning |
|---|---|---|
| `BENCH_MODE` | 1 | benchmark firmware rather than the interactive path |
| `HOT_POS` | 16 | fp32 hot window in internal SRAM |
| `WEIGHTS_PSRAM` | 0 | int8 weights in internal SRAM |
| `DUMP_LOGITS` | 1 | emit the nine-depth logit dumps |
| `KVCHECK` | 1 | run the accuracy harness |
| `VAR_RUNS` / `VAR_TOKENS` | 10 / 12 | short-run variance block |
| `CTX_RUNS` / `CTX_LEN` | 3 / 512 | three 512-token sustained runs |
| `KV_INT8` | 1 | int8 KV cache |
| `DUAL_CORE` | 1 | split attention heads across both cores |
| `DMA_PREFETCH` | 0 | no explicit DMA (a measured negative result) |

Each row of the paper's Table IV changes exactly one of these or one kernel
path, and each row was rebuilt and reflashed separately. `logs/BUILD-MAP.md`
maps every row to its capture.

## Cache variants

The 64 KB, 32 B line, eight-way configuration is the paper's primary build and
is selected by `config/sdkconfig.64kb` in the Build section above. The default
`sdkconfig.defaults` and a fresh `idf.py set-target esp32s3` produce the 32 KB
variant, which is the baseline for the cache sweep.

Switch either configuration with the helper script:

```bash
python3 tools/set_cache_cfg.py 32 32   # 32 KB, 32 B lines (baseline)
python3 tools/set_cache_cfg.py 64 32   # 64 KB, 32 B lines (primary)
```

Or copy `config/sdkconfig.32kb` or `config/sdkconfig.64kb` over `sdkconfig`.
Rebuild and reflash after changing the cache; the cache is programmed at
startup, so there is no runtime switch.

## Recomputing the tables

```bash
python3 tools/analyze_bench.py logs/bench_kvint8_final3.log
python3 tools/analyze_cache.py logs
```

`analyze_bench.py` writes a `.summary.json` next to the log. The per-build
summaries in `logs/` were produced by this script.

## Regenerating the figures

```bash
python3 tools/make_figures.py
```

The generator reads `logs/` and writes `logs/figs/`. It raises an error if a
required measurement is absent; it never substitutes a default value.

## Vendor microbenchmark

`micro/` is a separate ESP-IDF application that links ESP-DSP v1.8.2 and times
`dsps_dp_s8` against the same-tree kernels. Clone the measured tag:

```bash
git clone --depth 1 --branch v1.8.2 \
    https://github.com/espressif/esp-dsp.git ../components/esp-dsp
. $IDF_PATH/export.sh
cd micro
idf.py set-target esp32s3
idf.py build
```

`CONFIG_DSP_OPTIMIZED` must be `y`. Flash the app only at `0x10000`, capture
with the marker `MICRO done`, then:

```bash
python3 tools/micro_report.py
```

The report script reads both captures, fails if they disagree by more than
0.5%, and regenerates `logs/MICRO-RESULT.md`. The app prints the ESP-DSP tag
it was built against; it should be `v1.8.2`.

One check is one `CHECK` assertion. Each kernel run prints 22 `CHECK` lines:
both implementations at six positions on the 16-element score shape (12
checks), both at lengths 32, 128 and 496 (6 checks), and four positions of the
batched kernel (4 checks). The aggregate `MICRO correctness PASS` line
summarises those 22 assertions; it is not a 23rd check. A run that shows 23
`PASS` strings therefore has 22 checks plus the summary.

This microbenchmark is instruction-level and cache-resident. It does not
measure the PSRAM scan, and it does not test the ESP-DL Attention module.

The fused-load A/B is separate from the microbenchmark. Flash the v1.1.0 binary
(`bench_ab_base_1.log` and `_2.log`) and the v1.2.0 binary
(`bench_kvint8_fused_1.log` through `_6.log`) alternately in one session, with
the same `bench_cfg.h` and prompt. `tools/micro_report.py` recomputes the
medians and fails if the 42 logit blocks are not byte-identical.

## What is not reproduced here

- Energy per token. No shunt was attached to the board.
- Task-level accuracy. The accuracy study is logit-level, at nine depths.
- Other boards, models or context lengths. The paper reports one of each.

## Known variance

Sustained throughput across the six captures of the v1.2.0 build spans
168.244 to 168.282 tokens/s, about 0.023%. The board is reflashed between
captures, so that spread covers flash, boot and allocator placement, not only
run-to-run noise. The two v1.1.0 baseline captures used in the A/B land at
166.74 tokens/s.
