# Prequantized model library: blob format and swap state machine

Status: implemented in `v1.3.0`. This document records the original
architecture, design decisions, implementation details and validation results.
The remaining untested cases are listed in §13.

Baseline measured on 2026-11-25:

- SD: 1-bit, 40 MHz
- FATFS `open`/`read()`, 64 KiB chunks
- Sequential: **4.36 MB/s**
- Random 35 KiB: **4.08 MB/s** (~8.8 ms)
- FATFS mount: **~50 ms**
- Current active model: **274,656 B** of int8 weights + scales
- Free internal SRAM after placement: **~43 KB** on the bench build
  (placement is dynamic; see 4.1)
- Free PSRAM: **~6.9 MB**

These are the design-time figures. The release re-measured the same card in
`docs/RELEASE-v1.3.0.md`: FATFS reads 4.31 MB/s and a mount costs 82-88 ms.
Where the two disagree, the release figures are the ones behind the v1.3.0
timings.

The goal is a cold-path model library: many models on SD, exactly one active in
SRAM, and a switch that is fast enough to feel like a session change rather
than a reboot.

## 1. Decisions

| Decision | Choice |
|---|---|
| Where models live | SD card, FATFS, `models/` directory |
| Where the active model executes | Internal SRAM, existing int8 arena |
| Where models are staged | PSRAM |
| Where inactive models are cached | PSRAM, three LRU slots |
| Last selection | NVS namespace `modellib`, basename only |
| Format | Prequantized, self-describing `.m8` blob |
| On-device quantization | Removed from the switch path |
| Runtime shape changes | Rejected in v1; same-shape models only |
| Tokenizer | Embedded in the blob, parsed and checked |
| SD during inference | Unmounted |
| Fallback | Flash-partition model at `0x210000` |
| Raw SD partition | Not in v1; only ~12% over FATFS `read()` |
| Compression | Not in v1; int8 weights are likely near-incompressible |
| Per-token expert loading from SD | Out of scope; ~4.4 MB/s makes it non-viable |
| MoE experts | PSRAM-resident FFN sidecar, copied into the SRAM arena; §12 |

The important separation is:

- **SD** = cold library
- **PSRAM** = staging and the inactive-model cache
- **SRAM** = the only execution tier

## 2. Blob format

### 2.1 Naming and layout

```text
/sdcard/models/
  default.m8
  stories260k-q8.m8
  commands-q8.m8
  ...
```

Each `.m8` file is self-describing. No separate sidecar is required for v1. A
directory scan reads the first 4 KiB of each file to build the list.

Keep filenames reasonably short. The main firmware has long-filename support
enabled, but a short filename is more robust if the card is prepared on another
system.

### 2.2 Header

All fields are little-endian. The header occupies the first 4 KiB of the file so
the payload starts on a 4 KiB boundary and every 16-byte row is naturally
aligned. The layout below is the wire format; the loader uses explicit offsets
and never assumes compiler padding.

```c
#define M8_VERSION      1
#define M8_HEADER_BYTES 4096
#define M8_MAX_TENSORS  16
#define M8_MAX_BYTES    (512 * 1024)
#define M8_TENSOR_OFF   256
#define M8_TENSOR_BYTES 36
#define M8_TIER_ANY     2

// header_crc32 and sanity_logits_crc32 are the standard reflected CRC-32
// (poly 0xEDB88320) with init and xorout 0xFFFFFFFF. That is exactly
// esp_rom_crc32_le(0, buf, len) on this target, because the ROM wrapper
// inverts at entry and exit; it is also exactly zlib.crc32(). Python
// reference: crc32() in tools/logit_dump.py. The non-inverting "init 0, no
// xorout" variant does NOT match the device.
```

| Offset | Size | Field | Notes |
|---:|---:|---|---|
| 0 | 4 | `magic` | `"M8B1"` |
| 4 | 2 | `version` | 1 |
| 6 | 2 | `header_bytes` | 4096 |
| 8 | 4 | `file_bytes` | header + payload |
| 12 | 4 | `payload_bytes` | includes trailing pad to 4 KiB |
| 16 | 4 | `flags` | bit 0: tokenizer embedded |
| 20 | 4 | `header_crc32` | over `[0,4096)` with this field zeroed |
| 24 | 32 | `payload_sha256` | over `[4096,file_bytes)` |
| 56 | 2 | `dim` | |
| 58 | 2 | `hidden_dim` | |
| 60 | 2 | `n_layers` | |
| 62 | 2 | `n_heads` | |
| 64 | 2 | `n_kv_heads` | |
| 66 | 2 | `vocab_size` | |
| 68 | 2 | `seq_len` | |
| 70 | 2 | `ctx_default` | |
| 72 | 1 | `quant_bits` | 8 |
| 73 | 1 | `row_align` | 16 |
| 74 | 2 | `tensor_count` | <= 16 |
| 76 | 4 | `q_bytes` | total int8 payload |
| 80 | 4 | `scale_bytes` | total fp32 scale payload |
| 84 | 4 | `norm_bytes` | total fp32 RMS payload |
| 88 | 4 | `tokenizer_offset` | relative to payload |
| 92 | 4 | `tokenizer_bytes` | |
| 96 | 1 | `sanity_prompt_len` | 0 = skip the sanity check |
| 97 | 1 | `sanity_topk` | usually 32 |
| 98 | 2 | `reserved0` | 0 |
| 100 | 4 | `sanity_logits_crc32` | see below |
| 104 | 16 | `sanity_prompt_ids[8]` | u16 token ids |
| 120 | 16 | `sanity_ids[8]` | u16 expected greedy ids |
| 136 | 4 | `reserved1` | 0 |
| 140 | 32 | `name` | NUL-padded display name |
| 172 | 32 | `build_id` | NUL-padded packer id |
| 204 | 52 | `reserved2` | 0 |
| 256 | 576 | `tensors[16]` | 16 x 36-byte entries |
| 832 | 3264 | padding | 0 |

Tensor entry:

| Offset | Size | Field | Notes |
|---:|---:|---|---|
| 0 | 2 | `id` | 1..11 |
| 2 | 2 | `flags` | 1 = Q8, 2 = F32, 4 = required |
| 4 | 4 | `rows` | |
| 8 | 4 | `cols` | |
| 12 | 4 | `stride` | q row stride in bytes, or 4*cols for F32 |
| 16 | 4 | `data_offset` | relative to payload, 16-aligned for Q8 |
| 20 | 4 | `data_bytes` | `rows*stride` for Q8, `rows*cols*4` for F32 |
| 24 | 4 | `scale_offset` | 0xFFFFFFFF when there are no scales |
| 28 | 4 | `scale_bytes` | `rows*4` for Q8 |
| 32 | 1 | `tier` | 0 = SRAM, 1 = PSRAM, 2 = any (advisory) |
| 33 | 3 | `reserved` | 0 |

The sanity fingerprint is defined precisely so the host and the device compute
the same value:

1. Run the token ids in `sanity_prompt_ids[0..sanity_prompt_len)`.
2. Take the final logits vector.
3. Select the top `sanity_topk` entries by value, descending, ties broken by
   the lower index.
4. Serialise those values as little-endian float32, in that order, and apply the
   CRC-32 above.

`sanity_ids[0..n)` are the expected greedy continuation, compared step by step
after the sanity prompt. The blob currently carries 8 ids; the real check is the
CRC plus the first few generated ids, not a single top-1.

`tier` is advisory, not a placement command. Measured runs place the same
274,656 B of int8 weights either fully in internal SRAM (260,608 B of q data
plus 14,048 B of scales) or with one 10,240 B matrix in PSRAM, depending on heap
fragmentation at boot. Forcing a per-tensor tier would bake a boot-dependent
choice into the file. The loader instead uses the existing largest-first
allocator, logs the actual split, and checks the total byte counts from the
header.

### 2.3 Tensor IDs

| ID | Tensor |
|---|---|
| 1 | token embedding |
| 2 | Wq |
| 3 | Wk |
| 4 | Wv |
| 5 | Wo |
| 6 | W1 |
| 7 | W2 |
| 8 | W3 |
| 9 | attention RMS norm |
| 10 | FFN RMS norm |
| 11 | final RMS norm |

The eight int8 matrices keep the current packing: all layers are stacked in the
row dimension exactly as `llm_quantize()` does today. This lets the blob bind
directly to the existing `qmat_t` buffers.

RMS norms stay fp32 and are stored as `M8_T_F32` tensors.

### 2.4 Payload

```text
[4 KiB header]
[int8 q data, 16-byte aligned]
[fp32 scales, 4-byte aligned]
[fp32 RMS norms]
[tokenizer blob]
[pad to 4 KiB]
```

Rules:

- `data_offset` and `scale_offset` are relative to `header_bytes`.
- Every `q_offset` is a multiple of 16.
- `data_bytes` must equal `rows * stride`, with overflow-checked arithmetic.
- `scale_bytes` must equal `rows * 4` for an int8 tensor.
- The tokenizer range must be inside the payload and not overlap tensor data.
- `file_bytes` must be at most `M8_MAX_BYTES`.
- `payload_sha256` covers `[header_bytes, file_bytes)`.
- `header_crc32` covers `[0, header_bytes)` with the CRC field set to zero.
- RoPE tables are **not** stored. The firmware already rebuilds them from
  `dim`, `head_size`, `seq_len`, and `ctx`. A same-shape swap reuses the
  existing table.
- The tokenizer is embedded so a model and its tokenizer cannot be mismatched.
  The current `tok512.bin` is only 6,227 B. Its bytes are covered by
  `payload_sha256`, so no separate tokenizer hash is needed in v1; "tokenizer
  check" means the semantic checks below, not a second integrity hash.

Measured blob size (packed 2026-11-25):

```text
int8 weights + scales   274,656 B
RMS norms                 2,816 B
tokenizer                 6,227 B
payload padding           3,021 B
header + tensor table     4,096 B
----------------------------------
total                   290,816 B = 284.0 KiB
```

### 2.5 Quantization must match the firmware

The packer uses exactly the current on-device rule:

- per-row symmetric int8
- scale = max(abs(row)) / 127, or 1.0 for an all-zero row
- `inv = 1.0f / scale`, then `q = trunc(row * inv + copysign(0.5, row * inv))`
- clamp to [-127, 127]
- pad each row to a 16-byte stride with zeros
- fp32 scales
- **all arithmetic in float32**, so the bytes match the C loop exactly

The reference implementation is `qmat_fill()` in `firmware/main/llm.c`. The
host check `tools/quant_ref.c` contains the same routine and is compared
byte-for-byte against the Python packer.

If the packer differs, the logits differ. The acceptance test is that a packed
model reproduces the current int8 logits exactly.

### 2.6 Packer and verifier

Tools (all host-side, no firmware change):

| Tool | Purpose |
|---|---|
| `tools/pack_model_m8.py` | fp32 checkpoint + tokenizer -> `.m8`; computes CRC/SHA, sanity fingerprint, optional `--ref` and `--check` |
| `tools/quant_ref.c` | host copy of `qmat_fill()`, emits a reference byte stream for `--ref` |
| `tools/verify_m8.py` | independent reader: structure, CRC/SHA, dequant error bound, rebuilds the legacy model and runs `ref_llama.py` against a golden device dump |
| `tools/logit_dump.py` | parses `LOGD`/`LOGE` UART dumps, prints top-k, computes the sanity CRC, compares two dumps bit-for-bit |

The packer is deterministic: the same inputs produce a byte-identical blob
(checked inside `--check`).

Result for `stories260K.bin`:

```text
blob       /tmp/.../stories260k-q8.m8 (290,816 B)
sha256     a1ca0be5eae35261cac49c60056ed7fb7201b7328e3c5f4b987e098372c61306
sanity     prompt [1,403,407,261,378], top32 crc 0xb2f5d2bd
greedy ids  432 383 286 261 376 298 315 421 395 317 426 338
```

The rebuilt fp32 model gives the same top-5 and the exact same 12-token greedy
sequence as the device, and the host quantizer matches `quant_ref.c`
byte-for-byte. See `logs/M8-PACK-2026-11-25.txt`.

## 3. Discovery and manifest

v1 uses directory scan, not a manifest:

1. `opendir("/sdcard/models")`
2. for each `*.m8`, open with `open()`
3. read 4 KiB header with `read()`
4. check magic, version, header size, file size, dimensions, tensor table bounds
5. read only the embedded tokenizer (a few KiB) and CRC-32 it
6. append `{filename, name, bytes, dims, sanity CRC, tokenizer CRC}`

This avoids a manifest that can drift out of sync with the files. The header is
the source of truth. A full SHA-256 is verified only when a model is loaded, not
during listing, so a scan costs one 4 KiB read plus one tokenizer read per file.

`model_info_compat()` compares that list entry's config and tokenizer CRC with
the active model. It is a cheap hint for a UI: mismatched entries would need a
reboot. The authoritative check still happens in `model_compat()` at apply time,
including every tensor row/col/stride and a byte comparison of the tokenizer.

## 4. Runtime memory plan

### 4.1 Active arena

The current firmware allocates the int8 weights as eight `qmat_t` buffers, with
the largest-first fallback. The library should keep that arena alive for the
life of the process.

Measured on the current build (2026-11-25 and the restored boot log):

```text
q data          260,608 B   internal SRAM
scales           14,048 B   internal SRAM
--------------------------------------------
int8 total      274,656 B   free internal after load ~43 KB (bench build)
```

Earlier runs placed one 10,240 B matrix in PSRAM instead (250,368 B internal /
10,240 B PSRAM). Placement depends on heap fragmentation at boot, not on the
blob, which is why `tier` stays advisory. The loader must log the actual split
and fail loudly if the internal budget cannot be met.

For v1, runtime swap requires **exact shape match**:

- same `dim`, `hidden_dim`, `n_layers`, `n_heads`, `n_kv_heads`, `vocab_size`
- same `rows`, `cols`, `stride` for every tensor
- same tokenizer vocabulary

If the config or tensor layout differs, return `MODEL_ERR_SHAPE`; a different
embedded tokenizer returns `MODEL_ERR_TOKENIZER`. Different shapes are a
boot-time choice, not a runtime swap.

This restriction is what lets the swap be a set of `memcpy()` calls into
existing buffers, with no allocator activity and no fragmentation risk.

### 4.2 Staging

The blob is read into PSRAM first, validated, and only then copied into SRAM.
The staging buffer is allocated for the exact file size (290,816 B for the
current model) at each switch and freed after the copy. `M8_MAX_BYTES` is only
the validation ceiling; nothing reserves 512 KiB permanently.

Why stage instead of reading directly into SRAM:

- the old model remains intact until the new one is fully validated
- a short read or bad checksum does not destroy the active model
- the SD card can be unmounted before the SRAM copy
- the staging buffer becomes the new active blob after a successful switch

The committed active blob is kept in PSRAM, so a failed sanity check can restore
the previous weights without the card. In steady state the live set is the active
blob plus up to three cached blobs. A cold switch briefly adds the incoming stage
as a fifth blob, about 1.40 MiB transiently, and the old active is then either
cached or evicted. That is the rollback mechanism: the old arena is refilled in
place from the retained active stage, then its sanity fingerprint is re-checked.

### 4.3 PSRAM model cache

Implemented 2026-11-25. Three inactive validated blobs stay resident in PSRAM,
plus the active one:

```text
active             290,816 B
3 cached slots     884,748 B  (294,916 B each, including heap overhead)
------------------------------
steady total     1,175,564 B  (~1.12 MiB)
cold transient   1,470,480 B  (~1.40 MiB, with the incoming stage)
```

The cache moves, never copies, stages between the active slot and a cache slot.
Every successful `model_apply()` deposits the model it replaced into the cache,
so switching back is immediate. Replacement is LRU; a slot is also reused
directly when the same path is stored again.

Measured on device (`logs/M8-LIB-2026-11-25.log`, `logs/M8-LIB-LRU-2026-11-25.log`):

| Path | Time |
|---|---:|
| cold (mount + read + SHA + copy + sanity) | 377–405 ms |
| cached (copy + sanity) | 29–30 ms |

PSRAM after one cached blob: 6,761,312 B free. With all three slots populated:
6,171,480 B free. The difference is exactly three 294,916 B allocations, so the
budget tracks the design and nothing leaks across evictions.

### 4.4 State to reset on swap

A model swap is a new session. After copying the new weights:

- zero the KV cache (or at least reset the position and overwrite as needed)
- reset the token position to 0
- reset per-forward profiling counters
- reset any sampler state
- keep the tokenizer object only if it matches

The KV cache is model-specific. Even with the same shapes, old K/V values must
not leak into the new model. Clearing 245 KiB of int8 KV is about 7 ms at the
measured PSRAM write rate.

## 5. Swap state machine

States are intentionally small. All transitions run in the control task; the
inference path is not re-entered until the machine returns to `ACTIVE`.

```text
BOOT
  -> MOUNT
  -> LOAD_DEFAULT
  -> ACTIVE

ACTIVE
  -> SWAP_MOUNT
  -> SWAP_HEADER
  -> SWAP_STAGE
  -> SWAP_VERIFY
  -> SWAP_COMPAT
  -> SWAP_QUIESCE
  -> SWAP_COPY
  -> SWAP_SANITY
  -> ACTIVE

Any failure after SWAP_QUIESCE
  -> ROLLBACK
  -> ACTIVE or FLASH_DEFAULT

Any failure before SWAP_QUIESCE
  -> ACTIVE (old model untouched)
```

| State | Entry action | Success event | Failure |
|---|---|---|---|
| `BOOT` | load flash default or `default.m8` | `ACTIVE` | flash default |
| `ACTIVE` | accept requests | `SWAP_MOUNT` on select | — |
| `SWAP_MOUNT` | mount SD at 40 MHz, 1-bit | `SWAP_HEADER` | `MODEL_ERR_IO`, stay active |
| `SWAP_HEADER` | open file, read 4 KiB, validate magic/version/size/CRC | `SWAP_STAGE` | reject, stay active |
| `SWAP_STAGE` | read payload into PSRAM staging with `read()` 64 KiB | `SWAP_VERIFY` | reject, stay active |
| `SWAP_VERIFY` | SHA-256 payload, check tensor bounds and tokenizer parse | `SWAP_COMPAT` | reject, stay active |
| `SWAP_COMPAT` | compare config, every tensor row/col/stride, and embedded tokenizer bytes with the active stage | `SWAP_QUIESCE` | `MODEL_ERR_SHAPE` / `MODEL_ERR_TOKENIZER`, stay active |
| `SWAP_QUIESCE` | caller guarantees inference is idle; no task switch | `SWAP_COPY` | caller contract (no `BUSY` error in v1) |
| `SWAP_COPY` | unmount SD; copy tensors into existing SRAM buffers | `SWAP_SANITY` | `ROLLBACK` |
| `SWAP_SANITY` | run fixed prompt; compare top-32 CRC and greedy ids | `ACTIVE` | `ROLLBACK` |
| `ROLLBACK` | refill the arena from the retained active PSRAM stage and re-check its fingerprint | `ACTIVE` | `FLASH_DEFAULT` |
| `FLASH_DEFAULT` | load `0x210000` model | `ACTIVE` | fatal error |

### 5.1 Measured timing

100 alternating switches on the S3-CAM, 40 MHz 1-bit SD (`logs/M8-SWITCH-2026-11-25.log`):

```text
mount SD at 40 MHz               88.9–89.5 ms
stage (read + SHA-256 + parse)   233.5–234.2 ms
copy PSRAM -> SRAM               4.70–4.71 ms
sanity (12 forwards)             17.02–17.06 ms
------------------------------------------------
total                            370.0–371.7 ms
```

`mount_us` is measured from the start of `sd_mount()` to its return. `stage_us`
covers the 284 KiB `read()` plus payload SHA-256 and structural validation; the
earlier 65 ms read estimate omitted the checksum. `sanity_us` is 5 prompt
forwards plus 7 greedy-token forwards. This is a session-change cost, not a
per-token cost.

The first switch after boot also pays the boot-time SD scan and partition seed
check; the table above is the steady state.

The previous raw legacy path was about **1.08 s** from `SD mounted` to model
availability, so the blob path is roughly 3× faster end to end.

### 5.2 Why the old model survives early failures

`SWAP_MOUNT`, `SWAP_HEADER`, `SWAP_STAGE` and `SWAP_VERIFY` never touch the SRAM
arena. A missing file, truncated blob, bad checksum or shape mismatch leaves the
active model fully usable. `model_compat()` then compares the staged config,
every tensor row/col/stride, and the embedded tokenizer bytes against the active
stage before anything is copied.

Only `SWAP_COPY` is destructive. On a sanity failure the previous weights are
restored from the retained active stage in PSRAM and that stage's own sanity
fingerprint is re-run. Rollback does not touch the card, so a bad blob or a card
removed mid-session cannot leave the device without a working model.

## 6. Implemented API (v2: library + cache)

```c
typedef struct {
    llm_t *m;
    tokenizer_t *tk;
    model_stage_t active;              // committed blob, PSRAM; rollback source
    model_stage_t cache[MODEL_CACHE_SLOTS];
    uint32_t cache_tick[MODEL_CACHE_SLOTS];
    uint32_t tick;
    bool bound;
} model_ctx_t;

typedef struct {
    uint32_t copy_us;
    uint32_t sanity_us;     // includes the rollback re-check on failure
    bool rolled_back;
} model_apply_stats_t;

int model_bind(model_ctx_t *ctx, model_stage_t *st, llm_t *m, tokenizer_t *tk);
void model_ctx_free(model_ctx_t *ctx);
int model_compat(const model_ctx_t *ctx, const model_stage_t *st);
int model_apply(model_ctx_t *ctx, model_stage_t *st,
                model_apply_stats_t *stats);

// Library: listing and cache
int model_probe_file(const char *path, model_info_t *info);
int model_scan_dir(const char *dir, model_info_t *out, size_t cap,
                   size_t *count);
int model_info_compat(const model_ctx_t *ctx, const model_info_t *info);
int model_cache_count(const model_ctx_t *ctx);
bool model_cache_has(const model_ctx_t *ctx, const char *path);
int model_cache_find(model_ctx_t *ctx, const char *path, model_stage_t *out);
int model_cache_put(model_ctx_t *ctx, model_stage_t *st);
void model_cache_remove(model_ctx_t *ctx, const char *path);
void model_cache_clear(model_ctx_t *ctx);
int model_apply_cached(model_ctx_t *ctx, const char *path,
                       model_apply_stats_t *stats);
```

`model_apply()` consumes `st` on every path and returns one of:

```text
MODEL_OK              new weights copied, sanity passed, stage committed
MODEL_ERR_SHAPE       config/tensor layout differs; old model untouched
MODEL_ERR_TOKENIZER   embedded tokenizer differs; old model untouched
MODEL_ERR_SANITY      copy happened but the fingerprint failed; rolled back
MODEL_ERR_ROLLBACK    rollback itself failed; caller must reboot
```

`model_apply()` is synchronous, allocates nothing, and is callable only when
inference is idle. The SD mount/read/unmount wrapper stays in the application,
so the library never depends on a mounted filesystem.

`model_apply_cached()` returns `MODEL_ERR_NOTFOUND` when the path is not
resident, which is the application's signal to mount the card, call
`model_stage_file()`, and then `model_apply()`. It never touches the filesystem
itself.

A scanned listing is advisory: `model_info_compat()` uses the config and the
tokenizer CRC, while `model_apply()` re-checks the full tensor table.

`model_prefs.c` stores the last successful selection in NVS under namespace
`modellib`, key `last`. It stores only the basename and never a `partition:*`
pseudo-path. NVS is initialized and deinitialized per call so its page cache does
not consume internal SRAM while the weights are live.

## 7. Failure handling

| Failure | Detection | Result |
|---|---|---|
| SD missing at boot | mount fails | load flash default |
| SD removed during read | `read()` error | reject, keep current |
| File missing | `open()` fails | reject, keep current |
| Bad magic/version | header | reject, keep current |
| Truncated file | `file_bytes` / short read | reject, keep current |
| Bad header CRC | CRC32 | reject, keep current |
| Bad payload SHA | SHA-256 | reject, keep current |
| Tensor offset out of range | bounds check | reject, keep current |
| Shape mismatch | dims/rows/stride | reject, keep current |
| Tokenizer mismatch | embedded tokenizer fails to parse, or its vocab/special-token layout does not match the header | reject, keep current |
| Copy failure | destination bounds | rollback |
| Sanity mismatch | top-32 logit CRC + greedy ids | rollback |
| Rollback failure | restored weights fail their own fingerprint | `MODEL_ERR_ROLLBACK`, caller reboots |
| Power loss during switch | reboot | flash default |
| Power loss during download | `.part` file incomplete | ignored; old file intact |

Download safety, when it is added:

1. write `model.m8.part`
2. flush and close
3. verify size and SHA-256
4. rename `model.m8.part` to `model.m8`
5. only then make it selectable

Do not run Wi-Fi downloads while inference is active. The Wi-Fi stack needs
internal SRAM, and the active model has only ~13 KiB of headroom. Use a
maintenance mode with the model unloaded, or write the card from a host.

## 8. Boot policy

Implemented 2026-11-25:

1. Mount SD at 40 MHz.
2. Seed `models/default.m8` from the flash partition if the file is missing.
3. Read the NVS last selection and try `models/<last>`.
4. If that is absent or invalid, try `models/default.m8`.
5. If that fails, load the flash-partition `.m8` blob at `0x210000`.
6. If that fails, fall back to the legacy checkpoint.
7. Persist the basename that actually loaded when it differs from the stored
   one. A partition fallback never overwrites the user's selection.

The NVS read and write are outside the inference path and outside the int8
arena. A stale or deleted last selection costs one failed open and then falls
through to `default.m8`.

## 9. Packer and validation tests

Host-side, completed 2026-11-25 (`logs/M8-PACK-2026-11-25.txt`):

| Test | Result |
|---|---|
| Pack `stories260K.bin` + `tok512.bin` | 290,816 B blob, 284.0 KiB |
| Header CRC32 and payload SHA-256 | pass |
| Every tensor offset/length inside the payload | pass |
| `q_offset % 16 == 0`, `scale_offset % 4 == 0` | pass |
| Quantizer vs `tools/quant_ref.c` | byte-identical (274,764 B) |
| Dequant error vs source fp32 | within half a quantisation step (ratio <= 1.0000) |
| Deterministic repack | byte-identical |
| Rebuilt legacy model vs device fp32 top-5 | match |
| Rebuilt legacy model greedy ids vs device int8 | exact 12/12 match |
| Corrupt payload byte | rejected: SHA-256 mismatch |
| Truncated file | rejected: file_bytes mismatch |
| Bad magic | rejected: bad magic |
| Blob size <= `M8_MAX_BYTES` | pass (290,816 <= 524,288) |

Golden device logits for the Step 2 comparison:

- `logs/golden-logits-2026-11-25.log` (fp32 and int8, 12 steps each)
- int8 step 0: top-1 = 432, top-5 = 432,383,322,353,323
- sanity fingerprint: prompt `[1,403,407,261,378]`, top-32 CRC `0xb2f5d2bd`
- int8 and fp32 greedy ids agree: `432 383 286 261 376 298 315 421 395 317 426 338`
- board restored to the exact standard image afterwards
  (`logs/restore_after_golden_2026-11-25.log`)

On device, completed 2026-11-25 (`logs/M8-BOOT-2026-11-25.log`):

| Test | Result |
|---|---|
| Stage `u32 len + blob` from the `model` partition to `/sdcard/models/default.m8` | pass, 290,816 B |
| Boot from the card with header CRC + payload SHA-256 + tensor bounds | pass |
| Embedded tokenizer semantic parse | pass, vocab 512, max token len 7 |
| Sanity top-32 CRC | pass, `0xb2f5d2bd` |
| Sanity greedy ids | pass, `432,383,286,...` |
| int8 logits vs golden, 12 steps x 512 values | **bit-identical (512/512 every step)** |
| Greedy continuation vs `EXPECTED` | 12/12 match |
| int8 placement | 260,608 B internal SRAM, 0 B PSRAM |
| Quantization at boot | skipped (prequantized) |
| Payload byte flipped | rejected: `payload SHA-256 mismatch` |
| Truncated by 1,000 B | rejected: `size mismatch` |
| Bad magic | rejected: `bad magic` |
| Tensor shape 512 -> 511 (header CRC fixed) | rejected: `unexpected shape 511x64` |
| Version 2 (header CRC fixed) | rejected: `unsupported version/header` |
| Tensor flags cleared (header CRC fixed) | rejected: `flags 0x4` |
| Tokenizer token length 5 -> 0x7ffffff0 (payload SHA + header CRC fixed) | rejected: `embedded tokenizer rejected` (`MODEL_ERR_TOKENIZER`) |
| Tokenizer rejection after the int8 arena is filled | arena freed, card remounted, legacy path boots |
| 10 consecutive boots from the partition with the card file ignored | 10/10 `sanity ok`, same CRC and placement (`logs/M8-SOAK-2026-11-25.txt`) |
| Every rejection falls back to the legacy checkpoint | pass |
| Board restored: standard image `4c96f65b...`, legacy model partition | pass |

On device, Step 3 runtime switch, completed 2026-11-25
(`logs/M8-SWITCH-2026-11-25.log`):

Model B is `tools/make_model_variant.py --scale 1.03`, a deterministic synthetic
variant of the same checkpoint, not a trained model. It has identical config,
tensor layout and embedded tokenizer, so `model_compat()` accepts it. Its
step-0 device dump is `logs/M8-ALT-GOLDEN-2026-11-25.log`; re-packing from that
file reproduces the blob sha256 byte for byte.

| Test | Result |
|---|---|
| Model B blob | 290,816 B, sha256 `f9265549...`, fingerprint `0x329da241` |
| B logits vs A golden, step 0 | all 512 values differ (max delta 1.294) |
| B greedy ids | coincide with A (uniform scaling is close to a logit rescale); the CRC still discriminates |
| Boot A, stage B from the `model` partition to `/sdcard/models/alt.m8` | pass |
| 100 alternating A <-> B switches at 40 MHz | 100/100 `rc=0`, `SWITCH_SUM ok=100 fail=0` |
| Sanity checks during the run | 51 x A `0xb2f5d2bd`, 50 x B `0x329da241` |
| Per-switch timing | mount 88.9-89.5 ms, stage 233.5-234.2 ms, copy 4.70-4.71 ms, sanity 17.02-17.06 ms, total 370.0-371.7 ms |
| Final active model and generation | A active, 12/12 golden ids (`SWITCH_GEN_MATCH=1`) |

Rollback test (`logs/M8-SWITCH-ROLLBACK-2026-11-25.txt`): one bit flipped in B's
`sanity_logits_crc32`, header CRC recomputed, payload SHA-256 left valid. The
host verifier still accepts the blob structurally, so the device is the only
check that catches it.

| Test | Result |
|---|---|
| Bad B staged and copied | pass; header CRC and payload SHA-256 are valid |
| Sanity check | rejected: `0x329da241 != 0x329da240`, `MODEL_ERR_SANITY` |
| Rollback | previous weights restored from the retained PSRAM stage, A re-verified at `0xb2f5d2bd` |
| Active model after rollback | A, build `packer1 b0a507e7ad0f`, 12/12 golden ids |
| Harness markers | `SWITCH_ROLLBACK_OK=1`, `SWITCH_GEN_MATCH=1` |
| Board restored: standard image `4c96f65b...`, legacy model partition | pass |

Compatibility test (`logs/M8-SWITCH-COMPAT-2026-11-25.txt`): one bit flipped in
B's embedded tokenizer score, payload SHA-256 and header CRC recomputed. The
blob stages successfully, then `model_compat()` rejects it before any copy.

| Test | Result |
|---|---|
| Staging a same-shape blob with a different tokenizer | pass |
| Compatibility check | rejected: `switch rejected (tokenizer)`, `MODEL_ERR_TOKENIZER` |
| Copy / sanity work | `copy=0`, `sanity=0`; the arena was never touched |
| Active model after rejection | A, 12/12 golden ids (`SWITCH_GEN_MATCH=1`) |

On device, model library (A+B), completed 2026-11-25
(`logs/M8-LIB-2026-11-25.log`, `logs/M8-LIB-NVS-2026-11-25.log`,
`logs/M8-LIB-LRU-2026-11-25.log`):

The test build provisions `/sdcard/models/alt.m8` from the partition, then
selects alt -> default -> alt. The first selection is cold, the next two hit the
PSRAM cache.

| Test | Result |
|---|---|
| `model_scan_dir()` on the card | 2 models listed, sorted, both `compat=0` |
| Header-only probe cost | 4 KiB header + 6,227 B tokenizer per file, no payload read |
| alt, cold | mount 88.3 ms, stage 233.9 ms, copy 4.7 ms, sanity 17.1 ms, total **377.6 ms** |
| default, cached | copy 4.6 ms, sanity 17.0 ms, total **29.4 ms** |
| alt, cached | copy 4.7 ms, sanity 17.0 ms, total **30.0 ms** |
| Generation after the sequence | 12/12 golden ids (`LIB_GEN match=1`) |
| End state | alt active, `cache=1/3`, PSRAM free 6,761,312 B |
| Reboot | `boot: using NVS last model alt.m8`, alt active, 12/12 golden ids |
| Select `default.m8`, persist, reboot | first pass 394.9 ms then restart; second boot `boot: using NVS last model default.m8`, `LIB_BOOT_CONFIRM default.m8` |

LRU test (`logs/M8-LIB-LRU-2026-11-25.log`), five files, same payload under
different names:

| Step | Source | Cache |
|---|---|---|
| alt, alt2, alt3 | disk each time | 1, 2, 3 slots |
| alt4 | disk; evicts `default.m8` | 3 |
| default | disk; evicts `alt.m8` | 3 |
| alt3, alt2 | **cache**, 30.1 / 29.6 ms | 3 |
| alt | disk; evicts `alt4.m8` | 3 |
| default | **cache**, 29.5 ms | 3 |

`LIB_SUM ok=8/8 cache=3 psram_free=6171480`: exactly active + three 294,916 B
blob allocations, with no growth across evictions. The standard image and the
legacy model partition were restored afterwards
(`logs/M8-LIB-RESTORE-2026-11-25.log`).

The cases this validation did not exercise are listed in §13.

## 10. Non-goals for v1

- Different architectures at runtime. Different shapes require a reboot in v1.
- Per-token expert loading from SD. The measured 4.4 MB/s read path cannot feed
  a per-token slice. PSRAM-resident experts are a separate experiment (§12).
- Executing the active model from PSRAM.
- Raw SD partition or custom block device.
- Compression.
- Model signing/authentication. SHA-256 is integrity only.
- Wi-Fi download during inference.

## 11. Recommendation

Build v1 in this order:

1. **Host packer** and header validator. **Done 2026-11-25** — blob, host C
   reference, independent verifier and negative tests all pass.
2. **Boot-time load** of one `.m8` blob, with logit comparison to the current
   path against `logs/golden-logits-2026-11-25.log`. **Done 2026-11-25** — the
   int8 dump is bit-identical for 12 x 512 values.
3. **Runtime switch** for same-shape models, with in-place refill, PSRAM
   rollback and flash fallback. **Done 2026-11-25** — 100/100 alternating
   switches, plus a forced sanity failure that rolled back cleanly.
4. **40 MHz + `read()`** in the loader. **Done 2026-11-25** — this is what the
   blob loader uses; the measured 284 KiB stage is 233 ms including SHA-256.
5. **Model library**: directory scan/list, header-only probe, NVS last-selected
   boot order. **Done 2026-11-25** — two-model listing, persisted selection
   across reboots.
6. **PSRAM model cache**: three LRU slots plus the active blob. **Done
   2026-11-25** — repeat switch 29–30 ms vs 377–405 ms cold, verified with
   three-slot eviction.

Step 4 can be done independently of the blob and is the lowest-risk firmware
change: same model, same partition, just a different read API and clock.

The raw-sector path is not worth the complexity for v1: it buys about 12%
(4.91 vs 4.37 MB/s) and requires a dedicated partition or unsafe assumptions
about FAT cluster layout.

The MoE path stays separate. The measurements say SD is a cold store; per-token
expert fetches from the card are not viable. PSRAM-resident FFN experts were
measured separately (§12): a pinned expert is free after one refill, natural
routing costs about 1.9x a plain forward at short context, and switching every
layer on every token costs about 3.3x. Quality remains untested because there is
still no trained MoE.

## 12. Synthetic MoE prototype (2026-11-25)

Status: mechanism and cost experiment. This host has no `torch`, no GPU and no
training script, so there is no trained MoE and this section makes no quality or
routing claim. The goals are narrower:

1. prove that an FFN slice can be moved into the live arena byte-exactly;
2. measure what per-token expert movement costs on this board;
3. keep the work isolated from the frozen `v1.2.0` research artifact.

### 12.1 What moves and what stays

- **Stays**: token embedding, attention, RMS norms and the tokenizer come from
  the active `.m8` model.
- **Moves**: each expert is only the FFN (`w1`, `w2`, `w3`) for all layers,
  stored in PSRAM in the same int8 row-strided form as the arena.
- **Hook**: `llm_forward_int8()` calls `moe_hook(m, layer, xb)` immediately
  after the FFN RMSNorm, before `quant_vec()` and `mm_q_pre()`. The int8 path
  only; the fp32 reference path is untouched.
- **Refill**: when the routed expert for a layer differs from the one resident,
  six `memcpy()` calls replace that layer's row range in `m->w1/w2/w3` and
  their scale rows. Shapes, pointers and row strides never change.
- **Arena**: still 274,656 B internal SRAM. Nothing is allocated inside a
  forward pass.

Because a layer is refilled before its matvecs, the router reads exactly the
normalized input the FFN is about to use.

### 12.2 Sidecar format

Little-endian, 4 KiB header, magic `MOE1`. It follows the `.m8` conventions:
header CRC32, payload SHA-256, explicit sizes and offsets.

| Offset | Field |
|---:|---|
| 0 | magic `MOE1` |
| 4 | `u16` version (1), `u16` header bytes (4096) |
| 8 | `u32` file bytes, `u32` payload bytes |
| 16 | `u32` flags (bit0 router f32, bit1 synthetic) |
| 20 | `u32` header CRC32 (field zeroed during the check) |
| 24 | `u8[32]` payload SHA-256 |
| 56 | `u16` dim, hidden, layers, experts |
| 64 | `u8` row align (16), 3 reserved bytes |
| 68 | `u32` sanity CRC of expert 0's source model |
| 72 | `u32` router offset, `u32` router bytes |
| 80 | `u32` expert block bytes, `u32` expert block offset |
| 88 | six `u32` pairs: `w1q`, `w1s`, `w2q`, `w2s`, `w3q`, `w3s` offset/bytes |
| 136 | `char[32]` name, `char[32]` build id |

The router is float32 `[layer][expert][dim]`. Each expert block contains whole
tensors (`w1q`, `w2q`, `w3q`, then the three scale arrays), so a per-layer
update is a row-range `memcpy()` per tensor.

Test fixture (`tools/make_moe_sidecar.py`):

```text
dim=64 hidden=172 layers=5 experts=2
router        2,560 B
expert block 174,560 B  (w1q 55,040 + w2q 56,320 + w3q 55,040
                         + scales 8,160)
file          355,776 B  sha256 55405dd3...
```

Expert 0 is the FFN of `default.m8`; expert 1 is the FFN of the synthetic
`alt.m8` (uniform scale 1.03). Expert blocks are extracted from packed `.m8`
files, not re-quantized, so the bytes match the arena exactly. Those blobs come
from the existing flow: A is the published `stories260k-q8.m8`, B is
`make_model_variant.py --scale 1.03` packed and fingerprinted as in section 9.

```bash
tools/make_moe_sidecar.py \
    --blob build/stories260k-q8.m8 --blob /tmp/alt.m8 \
    --out moe.bin --seed 2026 --check
```

### 12.3 Device results

`MOE_TEST=1`. The sidecar was provisioned from the flash model partition to
`/sdcard/models/moe.bin`, the backbone was booted normally, the sidecar was read
from SD and the card was unmounted before inference. Two full runs produced
identical CRCs, route counts and copy counts
(`logs/MOE-PROTOTYPE-2026-11-25.log`).

| Test | Result |
|---|---|
| Sidecar from SD | 355,776 B, 2 experts, header CRC and SHA-256 pass |
| Expert 0 forced | logit CRC `0x4a4a3c07` **equals** the no-MoE baseline |
| Expert 0 generation | 12/12 golden ids, same id CRC as baseline |
| Expert 1 forced | logit CRC `0x2e601e3d`, differs from baseline |
| Expert 1 repeat | 0 refills, same CRC, deterministic |
| Natural router | 56/29 expert calls over 85 layer-calls; 34 refills |
| Round-robin | 200 refills over 200 layer-calls (every layer, every token) |
| PSRAM after unbind | 7,056,228 B free, i.e. back to the pre-stage value |

Per-forward cost, replaying the 5-token prompt from position 0 (40 measured
forwards per case):

| Case | us/forward | vs base | Refills/forward |
|---|---:|---:|---:|
| No MoE | 1,266 | 1.00x | 0 |
| Pinned expert (router runs) | 1,266 | 1.00x | 0 |
| Natural router | ~2,430 (est.) | ~1.9x | 2.0 |
| Round-robin, every layer switches | 4,222 | 3.33x | 5.0 |

One refill moves 34,912 B (one layer: w1+w3+w2 plus scales) in about 588 us,
i.e. 58-60 MB/s out of PSRAM. A full five-layer switch costs 174,560 B and
about 2.95 ms per token.

### 12.4 Interpretation

- The refill path is exact: forcing expert 0, which is the booted model's own
  FFN, reproduces the baseline logits bit for bit. That validates the sidecar
  offsets, strides and copy ranges.
- A pinned expert costs only the router dot products: 0-2 us per forward. The
  one-time refill is 2.95 ms.
- Per-token routing is expensive at short context: 1.9x for this synthetic
  router and 3.3x in the worst case. At longer contexts the compute component
  grows, so the relative penalty falls, but the ~0.59 ms per refilled layer does
  not.
- Expert 1's greedy continuation coincides with expert 0's because the source
  variant is a uniform 1.03 scaling of the whole checkpoint. The logit CRC still
  distinguishes them; the id CRC is not a useful discriminator for this fixture.

Limits:

- Experts and router are synthetic. No load-balancing, perplexity or quality
  claim can be made.
- Two experts, top-1 routing, one expert resident per layer. Top-2 or larger
  expert counts need different copy and compute accounting.
- The sidecar stores whole expert tensors, 174,560 B each. Four experts would be
  about 698 KiB + router, which still fits, but it competes with the three-blob
  `.m8` cache for PSRAM.
- Per-token expert paging from SD remains out of scope; the measured card read
  rate is far too low.

The next useful refinement is session-level per-layer pinning: route once,
pay one 2.95 ms refill, then run without further copies. That is a trained-model
experiment, not a firmware one.

## 13. Untested cases

These are the cases the v1.3.0 validation did not exercise. They are listed so
the release is not read as more complete than it is.

1. **Card removal mid-read.** This needs physical access to the slot, so it
   remains a manual test. The missing-card and corrupt-file paths are covered,
   and the partition fallback is covered by the 10-boot soak.
2. **Power cycle during a switch.** §7 gives the intended behaviour (the next
   boot falls back to the flash default), but no real power cut was injected
   during staging, copy or sanity.
3. **A real different-shape blob on device.** `model_compat()` rejects a
   header-rewritten shape mismatch, but exercising the check with a second
   trained checkpoint needs that checkpoint. The host-side rules are covered by
   the header and tensor-table validators.
4. **NVS erased or full.** A stale selection falls through to `default.m8` by
   design; the deleted-namespace and out-of-space paths were not run on the
   device.
5. **Atomic download for models written over Wi-Fi.** The `.part` + verify +
   rename sequence is specified in §7, but Wi-Fi download is not implemented,
   so the sequence itself is untested.
