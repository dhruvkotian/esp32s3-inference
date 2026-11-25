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

    int dim, hidden, layers, heads, kv_heads, vocab, seq, ctx;
    int tensor_count;
    m8_tensor_t tensors[M8_MAX_TENSORS];

    uint32_t q_bytes, scale_bytes, norm_bytes;
    uint32_t tok_off, tok_bytes;

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

// Copy a staged blob into *m and *tk: allocate the run state and int8 arena,
// copy the prequantized weights and RMS norms, parse the embedded tokenizer,
// and run the sanity fingerprint. On failure *m and *tk are left empty.
int model_commit(model_stage_t *st, llm_t *m, tokenizer_t *tk);

// model_stage_file() + model_commit() + model_stage_free().
int model_load_m8(const char *path, llm_t *m, tokenizer_t *tk);
