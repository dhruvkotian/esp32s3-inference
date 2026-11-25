# Prequantized model library: blob format and swap state machine

Status: design plus host-side packer/verifier. Firmware is unchanged; the
loader and swap state machine are still unimplemented and need approval.

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

The goal is a cold-path model library: many models on SD, exactly one active in
SRAM, and a switch that is fast enough to feel like a session change rather
than a reboot.

## 1. Decisions

| Decision | Choice |
|---|---|
| Where models live | SD card, FATFS, `models/` directory |
| Where the active model executes | Internal SRAM, existing int8 arena |
| Where models are staged | PSRAM |
| Format | Prequantized, self-describing `.m8` blob |
| On-device quantization | Removed from the switch path |
| Runtime shape changes | Rejected in v1; same-shape models only |
| Tokenizer | Embedded in the blob, parsed and checked |
| SD during inference | Unmounted |
| Fallback | Flash-partition model at `0x210000` |
| Raw SD partition | Not in v1; only ~12% over FATFS `read()` |
| Compression | Not in v1; int8 weights are likely near-incompressible |
| Per-token expert loading | Out of scope; measured as non-viable |

The important separation is:

- **SD** = cold library
- **PSRAM** = staging and optional model cache
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
5. append `{name, filename, bytes, dims, sha-prefix}` to the list
6. close and continue

This avoids a manifest that can drift out of sync with the files. The header is
the source of truth. A full SHA-256 is verified only when a model is loaded, not
during listing.

An optional `models.idx` can be added later for a faster list, but it must not be
authoritative.

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

If any of these differ, return `MODEL_ERR_SHAPE`. Different shapes are a
boot-time choice, not a runtime swap.

This restriction is what lets the swap be a set of `memcpy()` calls into
existing buffers, with no allocator activity and no fragmentation risk.

### 4.2 Staging

Reserve one PSRAM staging buffer of `M8_MAX_BYTES` at boot. The blob is read
there first, validated, and only then copied into SRAM.

Why stage instead of reading directly into SRAM:

- the old model remains intact until the new one is fully validated
- a short read or bad checksum does not destroy the active model
- the SD card can be unmounted before the SRAM copy
- the staging buffer can become the rollback copy if desired

PSRAM cost: 512 KiB reserved, out of ~6.9 MB free. That is acceptable.

### 4.3 Optional model cache

Once the basic path works, keep the last N validated blobs in PSRAM:

- blob ~284 KiB
- 4 slots ~1.1 MiB
- 8 slots ~2.3 MiB

An LRU cache with the active model pinned makes a repeat switch a PSRAM copy
(~5 ms) plus sanity check, instead of a 50 ms mount + 65 ms read + checksum.

Start without the cache. Add it only if measured switch latency matters.

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
| `SWAP_MOUNT` | mount SD at 40 MHz, 1-bit | `SWAP_HEADER` | `MODEL_ERR_MOUNT`, stay active |
| `SWAP_HEADER` | open file, read 4 KiB, validate magic/version/size/CRC | `SWAP_STAGE` | reject, stay active |
| `SWAP_STAGE` | read payload into PSRAM staging with `read()` 64 KiB | `SWAP_VERIFY` | reject, stay active |
| `SWAP_VERIFY` | SHA-256 payload, check tensor bounds, shape, tokenizer | `SWAP_QUIESCE` | reject, stay active |
| `SWAP_QUIESCE` | ensure no inference, no worker, mark busy | `SWAP_COPY` | `MODEL_ERR_BUSY`, stay active |
| `SWAP_COPY` | unmount SD; copy tensors into existing SRAM buffers; parse tokenizer | `SWAP_SANITY` | `ROLLBACK` |
| `SWAP_SANITY` | reset state; run fixed prompt; compare top-1 IDs | `ACTIVE` | `ROLLBACK` |
| `ROLLBACK` | reload previous model from PSRAM cache, SD, or flash | `ACTIVE` | `FLASH_DEFAULT` |
| `FLASH_DEFAULT` | load `0x210000` model | `ACTIVE` | fatal error |

### 5.1 Timing estimate

```text
mount SD at 40 MHz              50 ms
read 4 KiB header               ~1 ms
read 284 KiB at 4.36 MB/s       65 ms
SHA-256 (estimate, to measure)  10–30 ms
unmount SD                      ~10 ms
copy PSRAM -> SRAM              ~5 ms
reset KV                        ~7 ms
sanity: 5 forwards              ~8 ms
------------------------------------------------
total                           ~156–206 ms
```

That is a session-change cost, not a per-token cost. If the blob is in the PSRAM
cache, the switch is roughly **20 ms**.

The current raw path is about **1.08 s** from `SD mounted` to model availability,
so this is a 5–9× improvement even before caching.

### 5.2 Why the old model survives early failures

`SWAP_MOUNT`, `SWAP_HEADER`, `SWAP_STAGE` and `SWAP_VERIFY` never touch the SRAM
arena. A missing file, truncated blob, bad checksum or shape mismatch leaves the
active model fully usable. Only `SWAP_COPY` is destructive, and it is preceded
by a validated staged copy and a rollback path.

## 6. API sketch

```c
typedef enum {
    MODEL_OK = 0,
    MODEL_ERR_MOUNT,
    MODEL_ERR_NOT_FOUND,
    MODEL_ERR_IO,
    MODEL_ERR_MAGIC,
    MODEL_ERR_VERSION,
    MODEL_ERR_SIZE,
    MODEL_ERR_CRC,
    MODEL_ERR_SHA,
    MODEL_ERR_SHAPE,
    MODEL_ERR_TOKENIZER,
    MODEL_ERR_BUSY,
    MODEL_ERR_SANITY,
    MODEL_ERR_ROLLBACK,
} model_err_t;

typedef struct {
    char     name[32];
    char     file[64];
    uint32_t bytes;
    uint16_t dim;
    uint16_t hidden_dim;
    uint16_t n_layers;
    uint16_t vocab_size;
    uint8_t  sha_prefix[8];
} model_info_t;

model_err_t model_lib_init(void);
model_err_t model_lib_scan(model_info_t *out, size_t cap, size_t *count);
model_err_t model_lib_switch(const char *name_or_file);
const model_info_t *model_lib_active(void);
```

`model_lib_switch()` is synchronous and callable only when inference is idle.
If a request arrives during a switch, return `MODEL_ERR_BUSY` rather than
blocking the inference task.

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
| Sanity mismatch | top-1 IDs | rollback |
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

v1:

1. Mount SD at 40 MHz.
2. Look for `models/default.m8`.
3. If present and valid, load it.
4. Otherwise load the flash-partition model at `0x210000`.
5. If that fails, stop with a clear error.

Optional later:

- store the last selected model name in NVS
- on boot, try last selected, then `default.m8`, then flash
- keep NVS writes out of the inference path

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

Still open:

1. Remove the card mid-read; needs physical access to the slot, so it remains a
   manual test. The partition path is already covered by the 10-boot soak.
2. Switch 100 times at 40 MHz; check for read errors and stable timing
   (Step 3, not yet approved).
3. Power-cycle during a switch; verify boot falls back cleanly (Step 3).

## 10. Non-goals for v1

- Different architectures at runtime. Different shapes require a reboot in v1.
- Per-token expert loading.
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
   path against `logs/golden-logits-2026-11-25.log`.
3. **Directory scan + list + switch** for same-shape models, with flash
   fallback and rollback.
4. **40 MHz + `read()`** in the loader. This is a standalone win even before
   the format exists: it cuts the current 1 MiB load from ~0.8 s to ~0.24 s.
5. **PSRAM model cache** only if switch latency needs to go below ~100 ms.

Step 4 can be done independently of the blob and is the lowest-risk firmware
change: same model, same partition, just a different read API and clock.

The raw-sector path is not worth the complexity for v1: it buys about 12%
(4.91 vs 4.37 MB/s) and requires a dedicated partition or unsafe assumptions
about FAT cluster layout.

The MoE path stays separate. The measurements say SD is a cold store; per-token
expert fetches would need a trained model and a very high SRAM hit rate before
they are worth prototyping.
