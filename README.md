# esp32s3-int8-transformer

Artifact for the paper **"Bytes, Not FLOPs: INT8 KV-Cache and Attention
Optimisation for Long-Context Transformer Inference on the ESP32-S3"**.

This repository is the experiment tree behind the paper. It runs a
260K-parameter decoder-only transformer at batch size one on an ESP32-S3, with
the key/value cache stored as int8 in external PSRAM and attention mapped onto
the 128-bit integer SIMD unit. At 512-token context the final build sustains
166.73 tokens/s, 2.872x the same-tree fp32 single-core baseline.

Every number in the paper comes from a raw UART capture in `logs/`. The mapping
from paper table rows to capture files is in [`logs/BUILD-MAP.md`](logs/BUILD-MAP.md).
Nothing in the figure pipeline is typed in by hand: `tools/make_figures.py`
fails if a required measurement is missing from a log.

## Hardware

| Item | Value |
|---|---|
| Board | ESP32-S3-CAM, 16 MB flash, 8 MB octal PSRAM |
| MCU | ESP32-S3, dual-core Xtensa LX7, 240 MHz, rev 0.2 |
| Internal SRAM | 512 KB |
| Data cache | 64 KB, 8-way, 32 B lines (32 KB variants in `config/`) |
| Host link | CH343 USB-UART bridge on `/dev/ttyACM0`, 115200 baud |
| Model storage | microSD, `/sdcard/stories260K.bin` and `/sdcard/tok512.bin` |

See [`docs/HARDWARE.md`](docs/HARDWARE.md) for the memory budget and the
measured PSRAM rates.

## Layout

```
firmware/      ESP-IDF project: attention kernels, int8 KV cache, dual core
config/        sdkconfig snapshots for the 32 KB and 64 KB cache variants
model/         stories260K.bin and tok512.bin, with provenance
data/          expected.json, the reference logits for the accuracy harness
logs/          raw UART captures, per-build summaries, figures, BUILD-MAP.md
tools/         capture, analysis, figure generation, cache configuration
micro/         separate app: ESP-DSP against the same-tree dot kernels
docs/          hardware notes and the reproduction procedure
```

## Quick start

```bash
. $IDF_PATH/export.sh
cd firmware
idf.py set-target esp32s3
idf.py build
```

Flash the application only. The bootloader, partition table and the model
partition are not touched:

```bash
esptool.py --port /dev/ttyACM0 --baud 921600 --chip esp32s3 write_flash \
  --flash_mode dio --flash_freq 80m --flash_size 16MB \
  0x10000 build/llm_s3_test.bin
```

Then capture one run:

```bash
python3 tools/capture_until.py /dev/ttyACM0 240 logs/bench_mine.log BENCH_DONE
```

The full procedure, including the build switches for each experiment, is in
[`docs/REPRODUCE.md`](docs/REPRODUCE.md).

## Vendor comparison

`micro/` is a separate application that links ESP-DSP v1.8.2 and times
`dsps_dp_s8` against `llm_dot16_s8`, `llm_dot16x4_s8` and `llm_dot_s8` on
internal-SRAM buffers. It exists to answer the obvious question: why not use
the vendor primitive? The answer is in [`logs/MICRO-RESULT.md`](logs/MICRO-RESULT.md).
On the score shape the batched kernel is 2.06x faster than four vendor calls;
on the long dot the vendor loop is 2.16x faster per block, but that gap is
worth at most 0.8% of the token because the V phase is memory-bound in situ.

## Regenerating the figures

```bash
python3 tools/make_figures.py
```

This reads `logs/` and writes `logs/figs/`.

## Verifying the release

```bash
sha256sum -c MANIFEST.sha256
```

The manifest covers every file in the release except itself. Regenerating a
figure will change the corresponding hash; that is expected, because PDF
producers embed timestamps.

## What this release does and does not reproduce

It reproduces the measurements: the logs are the captures the paper reports,
and the analysis scripts recompute every table and figure from them.

It does not reproduce the hardware by itself. Re-running the firmware needs an
ESP32-S3 with octal PSRAM, a microSD card, and the same ESP-IDF version
(v5.4.4). Throughput depends on the cache size and line size, so the cache
configuration is part of the experiment, not an implementation detail.

## Licensing

- Firmware, analysis scripts and build configuration: **GPL-3.0-only**, see
  [`LICENSE`](LICENSE).
- Manuscript source, figures and the text of this artifact: **CC BY 4.0**, see
  [`LICENSE-PAPER`](LICENSE-PAPER).
- `model/stories260K.bin` and `model/tok512.bin` come from
  [llama2.c](https://github.com/karpathy/llama2.c) under the MIT licence, see
  [`model/PROVENANCE.md`](model/PROVENANCE.md).

The two licences are deliberately separate. GPL-3.0-only covers the code; it
does not cover the paper.

## Citing

See [`CITATION.cff`](CITATION.cff).

## Scope

One board, one model, one context length, one memory configuration. The paper
makes no energy claim and no state-of-the-art claim. The comparison is against
same-tree baselines, which is the right comparison for attribution but not a
claim about the best possible implementation.
