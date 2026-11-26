# ESP32-S3 INT8 Transformer Inference

Firmware, kernels and measurement tooling for running a small int8 transformer
entirely on an ESP32-S3, plus a model runtime that loads and switches models
from an SD card.

The research baseline (`v1.2.0`) is the exact code and configuration behind the
TECS experimental claims: an int8 KV cache in PSRAM and 128-bit PIE attention at
512-token context. The runtime release (`v1.3.0`) adds the `.m8` model library,
a PSRAM cache, NVS boot selection and a synthetic MoE prototype. **`v1.3.0` was
not evaluated in the submitted manuscript.**

## Platform

| | |
|---|---|
| SoC | ESP32-S3, dual-core Xtensa LX7, 240 MHz, silicon v0.2 |
| Internal SRAM | 512 KB |
| PSRAM | 8 MB octal, 80 MHz |
| Flash | 16 MB |
| Model | 260K parameters, 512-token vocabulary, 512-token context |
| Toolchain | ESP-IDF v5.4.4 |

## Releases

| Tag | Role | Headline |
|---|---|---|
| `v1.2.0` | Research baseline | 166.6 decode forwards/s at 512 context; 2.898x the same-tree fp32 single-core baseline; 245,760 B int8 KV payload and scales |
| `v1.3.0` | Runtime evolution | `.m8` model library, three-slot PSRAM cache, NVS selection, synthetic MoE sidecar |

`v1.2.0` is the release that corresponds to the research results. `v1.3.0`
builds on it without changing the evaluated kernels or the measured claims.

## Layout

```
firmware/   ESP-IDF project: attention kernels, int8 KV cache, model runtime, tokenizer
micro/      standalone microbenchmark and ESP-DSP comparison
tools/      analysis, capture, packing, verification, reference and figure scripts
config/     sdkconfig snapshots for the 32 KB and 64 KB D-cache variants
data/       expected outputs and reference metadata
logs/       raw benchmark captures and runtime test logs
model/      model provenance, tokenizer and licences
docs/       hardware, reproduction, model library and SD benchmark documentation
```

## Quick start

The research configuration is the 64 KB D-cache build:

```bash
. $IDF_PATH/export.sh
cd firmware
idf.py set-target esp32s3
cp ../config/sdkconfig.64kb sdkconfig
idf.py build
idf.py -p /dev/ttyACM0 flash monitor
```

The 32 KB build is the cache-sweep baseline, not the headline configuration.

The model runtime tools build and verify `.m8` blobs:

```bash
python3 tools/pack_model_m8.py --help
python3 tools/verify_m8.py --help
python3 tools/make_moe_sidecar.py --help
```

## Results at a glance

All values are measured on the board above; the raw captures are in `logs/`.

| Metric | Value |
|---|---|
| Decode forwards/s at 512 context | 166.6 |
| Sampled tokens/s, same harness | 168.25 |
| Speed-up vs fp32 single core | 2.898x |
| Speed-up vs fp32 dual core | 2.109x |
| Decode forward time | 6.00 ms |
| int8 KV payload and scales (PSRAM) | 245,760 B |
| Complete hybrid cache incl. SRAM hot window | 266,240 B |
| Short-context rate, 16-position hot window | 615 tok/s |
| Sustained rate over 512 tokens | 46.0 tok/s |
| Model library, cold select | 377.6 ms |
| Model library, cached select | 29.4 ms |

Accuracy: top-1 predictions match the FP32 reference at all nine evaluation
depths; combined relative logit RMSE is 1.08% to 4.56%.

## Reproducing the results

The logs are the raw UART captures the results are computed from. The analysis
scripts recompute the tables and figures from those logs:

```bash
python3 tools/analyze_bench.py
python3 tools/make_figures.py
python3 tools/micro_report.py
```

Verify the tree against the release manifest:

```bash
sha256sum -c MANIFEST.sha256
```

Re-running the firmware needs an ESP32-S3 with octal PSRAM, a microSD card and
ESP-IDF v5.4.4. Throughput depends on cache size and line size, so the cache
configuration is part of the experiment, not an implementation detail.

## Scope and limitations

- The 260K checkpoint writes simple children's-story English. It is a reflex-tier
  runtime, not an assistant.
- 615 tok/s is the short-context rate. Sustained over 512 tokens the rate is
  46.0 tok/s.
- Energy per token was not measured; no energy claim is made.
- Runtime model switching requires identical tensor shapes and tokenizer.
- The MoE prototype is synthetic: two experts, no training, no quality or
  routing claim.
- Per-token expert paging from SD is not viable at the measured card rate.

## Citation

See `CITATION.cff`.

## Licence

- Firmware, tools and build configuration: **GPL-3.0-only**, see `LICENSE`.
- Bundled 260K model and tokenizer: **MIT**, see `model/LICENSE` and
  `model/PROVENANCE.md`.
- Documentation and figures: **CC BY 4.0**, see `LICENSE-PAPER`.
