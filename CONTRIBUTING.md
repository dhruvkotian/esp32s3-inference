# Contributing

This repository covers the firmware, kernels, model runtime, measurement tools
and documentation. The manuscript and its production files are maintained
separately and are out of scope here.

## Prerequisites

- ESP-IDF v5.4.4 at `$IDF_PATH`.
- ESP32-S3 with 8 MB octal PSRAM and 16 MB flash.
- A microSD card for the model-runtime path.
- Python 3 for the analysis and packing tools.

## Build

```bash
. $IDF_PATH/export.sh
cd firmware
idf.py set-target esp32s3
cp ../config/sdkconfig.64kb sdkconfig
idf.py build
```

`config/sdkconfig.64kb` is the research configuration; `config/sdkconfig.32kb`
is the cache-sweep baseline. The microbenchmark is a separate application:

```bash
cd micro
idf.py set-target esp32s3
idf.py build
```

## Measurement changes

A change that claims a performance difference needs:

1. The same build, cache configuration, core assignment and prompt as the
   baseline.
2. Raw UART captures committed under `logs/`, not a summary alone.
3. The exact command and marker used for the capture.
4. Run-to-run spread, not a single best run.
5. A statement of what did **not** move.

Do not add an energy claim without a measurement. Do not compare across
different cache or core configurations without saying so.

## Code

- C99, ESP-IDF style, no dynamic allocation inside a forward pass.
- Keep the int8 path and the fp32 reference path separate.
- Preserve 16-byte alignment for SIMD buffers.
- New `.m8` or `MOE1` fields must keep the header versioned and the CRC and
  SHA-256 checks valid.
- Do not commit `firmware/build/`, `sdkconfig`, or generated `__pycache__`.

## Commit and pull-request hygiene

- Keep a change focused; explain the measurement or correctness impact.
- Never commit credentials, Wi-Fi details, MAC addresses, private IPs or
  absolute home paths.
- Do not add manuscript PDFs, LaTeX sources or submission records.
- If a change belongs in a release, regenerate the manifest:

```bash
./tools/make_manifest.sh
sha256sum -c MANIFEST.sha256
```

## Licensing

Contributions to firmware, tools and build configuration are GPL-3.0-only.
Documentation and figures are CC BY 4.0. The bundled model and tokenizer remain
MIT; do not relicense them.
