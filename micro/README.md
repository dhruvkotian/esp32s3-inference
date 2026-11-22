# Vendor microbenchmark

This is a separate ESP-IDF application. It links ESP-DSP and times
`dsps_dp_s8` against the same-tree kernels on the shapes the attention scan
uses. It does not touch the paper firmware, and its results are in
[`../logs/MICRO-RESULT.md`](../logs/MICRO-RESULT.md).

## What it measures

| Case | Why |
|---|---|
| one 16-lane dot | the score path, one head per position |
| four positions, one call | `llm_dot16x4_s8`, the batched score path |
| four positions, four calls | the same work without batching |
| len 32, 128, 496 | the long dot used for the V accumulation |

All buffers are in internal SRAM, so the numbers are instruction cost. The
working set is 15,872 B, which fits the 64 KB data cache.

## Build

ESP-DSP is an external component. Clone the tag that was measured:

```bash
git clone --depth 1 --branch v1.8.2 \
    https://github.com/espressif/esp-dsp.git ../components/esp-dsp
```

Or point at an existing checkout:

```bash
export ESP_DSP_PATH=/path/to/esp-dsp
```

Then:

```bash
. $IDF_PATH/export.sh
cd micro
idf.py set-target esp32s3
idf.py build
```

`CONFIG_DSP_OPTIMIZED` must be `y`, which is the default. The app prints the
tag it was built against; check it matches the checkout.

## Run

Application only, at `0x10000`:

```bash
esptool.py --port /dev/ttyACM0 --baud 921600 --chip esp32s3 write_flash \
  --flash_mode dio --flash_freq 80m --flash_size 16MB \
  0x10000 build/micro.bin
python3 ../tools/capture_until.py /dev/ttyACM0 60 \
  ../logs/micro_dsp_vs_ours.log "MICRO done"
```

The app exits after printing `MICRO done`; there is no loop.

Restore the paper firmware afterwards from `../firmware` (same command, with
`build/llm_s3_test.bin`), or from the binary used for the paper.

## Report

```bash
python3 ../tools/micro_report.py
```

This reads the four microbenchmark captures (two per kernel build) plus the
eight end-to-end A/B captures, checks that two captures of the same kernel
agree to within 0.5%, and writes `../logs/MICRO-RESULT.md`. It also fails if
the v1.2.0 logits are not byte-identical to the v1.1.0 logits, so the report
cannot drift from the captures.

`main/dot_s8.S` is a byte-identical copy of `../firmware/main/dot_s8.S`, which
is the v1.2.0 fused kernel. The two-load rows in the report come from the
v1.1.0 copy captured before the fused load was adopted.
