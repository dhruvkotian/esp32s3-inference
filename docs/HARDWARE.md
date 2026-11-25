# Hardware

## Board

| Item | Value |
|---|---|
| Board | ESP32-S3-CAM generic carrier |
| Board vendor and revision | Not marked on the unit and not recorded |
| MCU | ESP32-S3, dual-core Xtensa LX7 |
| Clock | 240 MHz |
| Silicon revision | 0.2, efuse block revision 1.3 |
| Internal SRAM | 512 KB |
| External PSRAM | 8 MB, octal, 80 MHz, AP Memory, device ID 0x02 (generation 3), 64 Mbit, 32 B burst |
| PSRAM package order code | Not recorded |
| Flash | 16 MB, Boya, DIO mode, 80 MHz |
| Host link | CH343 USB-UART bridge, `/dev/ttyACM0`, 115200 baud |
| Model storage | microSD card, FAT, `/sdcard/stories260K.bin`, `/sdcard/tok512.bin` |

The board is a generic ESP32-S3-CAM carrier. The vendor, board revision and
module order code are not marked on the unit and were not recorded, so they are
not claimed. The PSRAM details are the ones reported by the ESP-IDF boot
probe. The camera is never initialised and is not part of any measurement. The
board is used as a compute and memory substrate.

## Memory budget in the final build

| Region | Bytes |
|---|---:|
| int8 weight matrices, internal SRAM | 250,368 |
| int8 weight matrices, PSRAM | 10,240 |
| Per-row weight scales | 14,048 |
| int8 K payload | 81,920 |
| int8 V payload | 81,920 |
| K scales | 40,960 |
| V scales | 40,960 |
| fp32 hot window, K and V | 20,480 |
| Free internal SRAM after placement | 13,231 |
| Largest free internal block | 7,680 |

The PSRAM-resident int8 cache is 245,760 B; with the 20,480 B fp32 hot window
it is 266,240 B in total. The cache lives in PSRAM because the weights and
working buffers leave 17,679 B free in internal SRAM, which is smaller than the
245,760 B PSRAM portion.

## Data cache

The ESP32-S3 programs its data cache at application startup, so the cache size
and line size are build-time choices and every variant is an app-only reflash.

| Variant | Size | Line | Associativity |
|---|---:|---:|---:|
| Final | 64 KB | 32 B | 8-way |
| Sensitivity | 32 KB | 32 B | 8-way |

`config/sdkconfig.32kb` and `config/sdkconfig.64kb` are the full sdkconfig
snapshots for the two variants. `tools/set_cache_cfg.py` edits a live
`sdkconfig` to a chosen size and line size.

The ESP32-S3's Kconfig encodes the cache size as a choice plus a numeric
symbol, and the 16 KB choice intentionally keeps `SIZE=0x8000` and occupies
half the cache at startup. `set_cache_cfg.py` sets both so the build stays
self-consistent.

## Measured PSRAM rates

| Pattern | MB/s |
|---|---:|
| Sequential read | 60.6 |
| Strided read, 16 of 32 B | 30.3 |
| Sequential write | 36.8 |

These are measured by the boot self-test in `firmware/main/main.c`, and the
raw output is in `logs/bench_kvint8_final3.log`. The attention scan achieves
51.7 MB/s of streamed KV traffic, about 85% of the sequential read rate.

## Timing

Two clocks appear in the logs:

- Wall time per token, from `esp_timer_get_time()`.
- Core time, from `esp_cpu_get_cycle_count()` divided by 240 MHz.

The four attention phases are core time summed over both cores, so they sum to
1.94x the attention wall time. This is expected and is not a bug.
