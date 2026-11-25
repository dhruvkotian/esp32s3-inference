# v1.3.0 — Model library, PSRAM cache, and a measured MoE prototype

Base: `v1.2.0` (the characterisation artifact). This release adds the model
runtime and a synthetic expert prototype. It does not change the paper.

The ACM TECS submission is **not** part of this repository or this release.

## Added

- **`.m8` model library.** Prequantized, self-describing blobs: 4 KiB header,
  embedded tokenizer, header CRC32, payload SHA-256, on-device sanity
  fingerprint. Header-only probe and directory scan.
- **PSRAM staging and cache.** Three inactive slots plus the active model;
  LRU eviction; `model_apply_cached()` swaps stages without copying the blob
  again.
- **NVS last-selected model** (`modellib`/`last`, basename only) with the boot
  order `default.m8` → last selection → `default.m8` → flash partition →
  legacy path.
- **Runtime switching** that rejects shape or tokenizer changes; those require
  a reboot.
- **Synthetic MoE prototype.** `MOE1` sidecar with FFN-only experts, staged
  from SD into PSRAM, with a per-layer row-range `memcpy()` refill into the
  existing internal-SRAM arena. Hook runs after the FFN RMSNorm in the int8
  path only.
- **Tools.** `pack_model_m8.py`, `verify_m8.py`, `make_model_variant.py`,
  `make_moe_sidecar.py`, `verify_submission_pdf.py`.
- **Docs and logs.** `docs/MODEL-LIBRARY-DESIGN.md`, `logs/MOE-*`,
  `logs/SD-BENCH-*`.

## Measured on the board

ESP32-S3-CAM, 240 MHz, 8 MB octal PSRAM, 16 MB flash, SD at 40 MHz 1-bit.

| Test | Result |
|---|---|
| Cold model select (SD → PSRAM → SRAM) | 377.6 ms |
| Cached select | 29.4 / 30.0 ms |
| LRU cache, 8 selects | 8/8 correct, 3 slots used |
| MoE, pinned expert | 1.00× a plain forward |
| MoE, natural synthetic router | ~1.9× |
| MoE, round-robin worst case | 3.33× (4,222 µs vs 1,266 µs) |
| Expert refill throughput | 58–60 MB/s (~588 µs per 34,912 B layer) |
| PSRAM after unbind | returns to the pre-stage value |

Forcing the booted model's own FFN reproduces the baseline logits bit for bit
(`crc 0x4a4a3c07`), which validates the sidecar offsets and copy ranges.

## Limits

- The MoE fixture is synthetic: two experts, no training, no load balancing.
  **No quality or routing claim is made.**
- Runtime switching requires identical tensor shapes and tokenizer.
- Per-token expert paging from SD is not viable at the measured card rate.
- Energy per token remains unmeasured.

## Compatibility

- Firmware: ESP-IDF v5.4.4, ESP32-S3 (tested on silicon v0.2).
- Model: `stories260K` and its 512-token tokenizer, MIT, as before.
- `v1.2.0` remains frozen and unchanged; this tag adds to it.
