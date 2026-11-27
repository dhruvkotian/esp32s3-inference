#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "llm.h"
#include "tokenizer.h"

// Wire-format constants. These must match tools/pack_model_m8.py.
#define M8_MAX_BYTES    (512 * 1024)
#define M8_HEADER_BYTES 4096
#define M8_MAX_TENSORS  16
#define M8_TENSOR_OFF   256
#define M8_TENSOR_BYTES 36

// Inactive models kept in PSRAM for fast switching. The active model is held
// separately, so peak PSRAM is (MODEL_CACHE_SLOTS + 1) blobs. Override with
// -DMODEL_CACHE_SLOTS=0 to measure the no-cache configuration; the loops that
// walk the cache become no-ops and the arrays are zero-length.
#ifndef MODEL_CACHE_SLOTS
#define MODEL_CACHE_SLOTS 3
#endif

typedef enum {
    MODEL_OK = 0,
    MODEL_ERR_IO = -1,
    MODEL_ERR_SIZE = -2,
    MODEL_ERR_MAGIC = -3,
    MODEL_ERR_VERSION = -4,
    MODEL_ERR_HEADER_CRC = -5,
    MODEL_ERR_SHA = -6,
    MODEL_ERR_SHAPE = -7,
    MODEL_ERR_TOKENIZER = -8,
    MODEL_ERR_NOMEM = -9,
    MODEL_ERR_SANITY = -10,
    MODEL_ERR_ROLLBACK = -11,
    MODEL_ERR_NOTFOUND = -12,
} model_err_t;

typedef struct {
    uint16_t id;
    uint16_t flags;
    uint32_t rows, cols, stride;
    uint32_t data_offset, data_bytes;
    uint32_t scale_offset, scale_bytes;
    uint8_t tier;
} m8_tensor_t;

typedef struct {
    // Whole file or partition image, owned. The header is at blob[0..4096).
    uint8_t *blob;
    size_t file_bytes;
    uint8_t *payload;       // blob + M8_HEADER_BYTES
    size_t payload_bytes;

    // Source path used as the PSRAM cache key. "partition:<label>" for a
    // length-prefixed flash partition, the FATFS path for a card file.
    char path[64];

    int dim, hidden, layers, heads, kv_heads, vocab, seq, ctx;
    int tensor_count;
    m8_tensor_t tensors[M8_MAX_TENSORS];

    uint32_t q_bytes, scale_bytes, norm_bytes;
    uint32_t tok_off, tok_bytes;
    uint32_t tok_crc;       // CRC-32 of the embedded tokenizer bytes

    int sanity_prompt_len;
    int sanity_topk;
    uint32_t sanity_crc;
    uint16_t sanity_prompt[8];
    uint16_t sanity_ids[8];

    char name[33];
    char build_id[33];
} model_stage_t;

// Human-readable form of a model_err_t, for boot logs.
const char *model_strerror(int rc);

// Read and validate a blob from a FATFS path. Uses open()/read() in 64 KiB
// chunks, so it gets the measured 4.36 MB/s instead of fread()'s 1.34 MB/s.
int model_stage_file(const char *path, model_stage_t *st);

// Read and validate a blob from a length-prefixed flash partition
// (u32 little-endian length, then the .m8 bytes). This is the fallback when
// the card is missing or the file is not there.
int model_stage_partition(const char *label, model_stage_t *st);

// Write a validated stage back to a FATFS path. Used once to seed the card
// from the flash partition; not part of the normal boot path.
int model_stage_save(const model_stage_t *st, const char *path);

void model_stage_free(model_stage_t *st);

// ---- Model library: probe and directory scan ------------------------------
//
// A probe reads only the 4 KiB header plus the embedded tokenizer, never the
// payload. That is enough to list a model and decide whether it could be
// swapped in without a reboot. The payload SHA-256 is deliberately skipped;
// model_stage_file() still verifies it before a model is applied.

typedef struct {
    char file[64];          // basename as listed in the directory
    char name[33];          // model name from the header
    char build_id[33];
    uint32_t file_bytes;
    uint32_t tok_crc;       // CRC-32 of the embedded tokenizer
    uint32_t sanity_crc;
    uint16_t dim, hidden, layers, heads, kv_heads, vocab, seq, ctx;
} model_info_t;

// Header-only probe of a .m8 file. No payload allocation, no SHA-256.
int model_probe_file(const char *path, model_info_t *info);

// List every valid *.m8 in `dir` as `model_info_t`, sorted by file name. `cap`
// bounds the output; *count receives the number stored. Invalid files are
// logged and skipped. Returns MODEL_OK even when the directory is empty.
int model_scan_dir(const char *dir, model_info_t *out, size_t cap,
                   size_t *count);

// Copy a staged blob into *m and *tk: allocate the run state and int8 arena,
// copy the prequantized weights and RMS norms, parse the embedded tokenizer,
// and run the sanity fingerprint. On failure *m and *tk are left empty.
int model_commit(model_stage_t *st, llm_t *m, tokenizer_t *tk);

// model_stage_file() + model_commit() + model_stage_free().
int model_load_m8(const char *path, llm_t *m, tokenizer_t *tk);

// ---- Runtime switch (v1: same shapes and tokenizer) ----------------------
//
// The active stage is kept in PSRAM after boot. It is both the compatibility
// reference and the rollback source, so a failed switch never needs the card
// again. *m and *tk must outlive the context.

typedef struct {
    llm_t *m;
    tokenizer_t *tk;
    model_stage_t active;
    // Inactive validated blobs, already resident in PSRAM. A switch to one of
    // these skips the SD mount, read and SHA-256 entirely.
    model_stage_t cache[MODEL_CACHE_SLOTS];
    uint32_t cache_tick[MODEL_CACHE_SLOTS];
    uint32_t tick;
    bool bound;
} model_ctx_t;

typedef struct {
    uint32_t copy_us;
    uint32_t sanity_us;      // includes the rollback re-check on failure
    bool rolled_back;
} model_apply_stats_t;

// Take ownership of a committed stage. Frees any previous active stage.
int model_bind(model_ctx_t *ctx, model_stage_t *st, llm_t *m, tokenizer_t *tk);

// Release the active stage. Does not touch *m or *tk.
void model_ctx_free(model_ctx_t *ctx);

// MODEL_OK when st has the same config, tensor layout and byte-identical
// embedded tokenizer as the active stage. Checked before anything is copied.
int model_compat(const model_ctx_t *ctx, const model_stage_t *st);

// Copy st into the live arena, run its sanity fingerprint, and make it the
// active stage. On sanity failure the previous weights are restored from the
// active stage and verified. Consumes st on every path. Call only when
// inference is idle; this is not thread-safe and does not allocate.
int model_apply(model_ctx_t *ctx, model_stage_t *st, model_apply_stats_t *stats);

// Cheap "could this model be swapped in?" hint for a scanned entry: compares
// the config and the tokenizer CRC. model_apply() repeats the authoritative
// check, including every tensor row/col/stride.
int model_info_compat(const model_ctx_t *ctx, const model_info_t *info);

// ---- PSRAM cache ----------------------------------------------------------
//
// Validated blobs are moved (never copied) between the active slot and the
// cache, so a cached switch costs a PSRAM -> SRAM copy plus the sanity check.
// Every successful model_apply() deposits the model it replaced into the
// cache, evicting the least recently used slot when all are full.

// Number of inactive blobs currently cached.
int model_cache_count(const model_ctx_t *ctx);

// True when `path` is resident in the cache.
bool model_cache_has(const model_ctx_t *ctx, const char *path);

// Move the cached entry for `path` into *out. Returns MODEL_ERR_NOTFOUND when
// it is not cached; *out is empty on every failure.
int model_cache_find(model_ctx_t *ctx, const char *path, model_stage_t *out);

// Take ownership of *st and cache it under st->path. Replaces an existing
// entry for the same path; evicts the least recently used slot when full.
int model_cache_put(model_ctx_t *ctx, model_stage_t *st);

// Drop the entry for `path`, if any. Freed immediately.
void model_cache_remove(model_ctx_t *ctx, const char *path);

// Drop every cached entry. Does not touch the active stage.
void model_cache_clear(model_ctx_t *ctx);

// Apply a model that is already in the PSRAM cache. Never touches the
// filesystem, so the application keeps ownership of SD mounting. Returns
// MODEL_ERR_NOTFOUND when `path` is not cached; the active model is untouched.
int model_apply_cached(model_ctx_t *ctx, const char *path,
                       model_apply_stats_t *stats);
