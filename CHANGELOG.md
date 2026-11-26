# Changelog

All dates are UTC. This repository is the public software tree; the manuscript,
its production files and submission records are maintained separately and are
not part of any commit here.

## v1.3.1 — 2026-11-26

Metadata and documentation patch. Firmware is byte-identical to `v1.3.0`; no
kernel, log, model or configuration file changes.

### Added

- `docs/RELEASE-PROVENANCE.md`: tag mapping and byte-for-byte verification of
  the public export against the private research tree.
- `docs/ARCHITECTURE.svg`: memory tiers, model library and execution diagram.

### Fixed

- `docs/MODEL-LIBRARY-DESIGN.md` records the implemented v1.3.0 state and lists
  the remaining untested cases in §13 instead of saying the loader is
  unimplemented.
- `README.md` separates the v1.2.0 research results from the v1.3.0 runtime
  measurements, and drops two short-context rates that describe an earlier build
  and are not in the v1.2.0 captures.
- `CITATION.cff`: current repository name and URL, the full university name, and
  a licence statement that matches `LICENSE-PAPER`.
- README and changelog label `2.898x` as a sampled-rate ratio against the
  58.06 tok/s baseline, so it is not read as applying to 166.6 forwards/s.
- MoE rows are labelled as per-forward latency ratios, and the natural-router
  figure is marked as an estimate.
- `CHANGELOG.md`: scan bandwidth corrected to 51.7 MB/s.

## v1.3.0 — 2026-11-25

Runtime evolution. Not evaluated in the submitted manuscript.

### Added

- `.m8` prequantized model format: 4 KiB header, magic `M8B1`, embedded
  tokenizer, header CRC32, payload SHA-256 and an on-device sanity fingerprint.
- Header-only probe and directory scan, PSRAM staging, and a three-slot inactive
  model cache with LRU eviction.
- `model_apply_cached()`: swap the active stage without copying the blob again.
- NVS last-selected model (`modellib`/`last`, basename only) and the boot order
  `default.m8` → last selection → `default.m8` → flash partition → legacy path.
- Runtime switching that rejects shape or tokenizer changes; those require a
  reboot.
- Synthetic MoE prototype: `MOE1` sidecar with FFN-only experts staged in PSRAM
  and a per-layer row-range refill into the internal SRAM arena.
- Tools: `pack_model_m8.py`, `verify_m8.py`, `make_model_variant.py`,
  `make_moe_sidecar.py`, `logit_dump.py`, `quant_ref.c`.
- Documentation: `docs/MODEL-LIBRARY-DESIGN.md`,
  `docs/SD-BENCH-2026-11-25.md`, runtime logs.

### Measured

| Test | Result |
|---|---|
| Cold model select | 377.6 ms |
| Cached select | 29.4 / 30.0 ms |
| LRU, 8 selects | 8/8 correct, 3 slots used |
| MoE pinned expert | 1.00x latency (no penalty) |
| MoE natural synthetic router | ~1.9x latency (~2,430 vs 1,266 us, estimated) |
| MoE round-robin worst case | 3.33x latency |
| Expert refill | 58–60 MB/s, ~588 us per layer |

Forcing the booted model's own FFN reproduces the baseline logits bit for bit.

### Limits

The MoE fixture is synthetic. No quality, routing or load-balancing claim is
made. Runtime shape changes still require a reboot.

## v1.2.0 — 2026-11-22

Research baseline. This is the release that corresponds to the TECS
experimental claims.

### Added

- Int8 KV cache in PSRAM: 245,760 B of payload and scales, 2.67x smaller than
  the 655,360 B fp32 cache; 266,240 B including the 20,480 B fp32 hot window in
  internal SRAM.
- Attention mapped to the 128-bit PIE unit, including the batched four-position
  K dot and the folded V-scale softmax.
- Fused long dot that matches ESP-DSP at 4.10 against 4.13 cycles per block and
  is worth 0.91% end to end.
- Dual-core split for the upper four heads.
- Standalone ESP-DSP v1.8.2 microbenchmark in `micro/`.
- Eighteen-run reproducibility check, accuracy sweep and negative results.

### Measured

| Metric | Value |
|---|---|
| Decode forwards/s at 512 context | 166.6 |
| Sampled tokens/s, same harness | 168.25 |
| Sampled-rate speed-up vs fp32 single core (58.06 tok/s) | 2.898x |
| Sampled-rate speed-up vs fp32 dual core (79.77 tok/s) | 2.109x |
| Decode forward time | 6.00 ms |
| Scan traffic | 238,080 B/token at 51.7 MB/s |

Top-1 predictions match the FP32 reference at all nine evaluation depths;
combined relative logit RMSE is 1.08% to 4.56%.
