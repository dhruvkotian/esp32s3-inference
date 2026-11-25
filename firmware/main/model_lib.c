// Boot-time loader for the prequantized .m8 model format.
//
// The format is produced by tools/pack_model_m8.py and documented in
// docs/MODEL-LIBRARY-DESIGN.md. Everything here is read-only with respect to
// the card and the flash: the blob is copied into the existing int8 arena and
// the embedded tokenizer is parsed into the usual tokenizer_t.
//
// Validation order is deliberate: magic/size/header-CRC first (cheap), then
// payload SHA-256, then tensor bounds and shapes, then the semantic tokenizer
// parse, and only then the on-device sanity fingerprint. A corrupt or
// truncated blob never reaches the allocator.

#include "model_lib.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_partition.h"
#include "esp_rom_crc.h"
#include "mbedtls/sha256.h"

static const char *TAG = "model";

#define M8_MAGIC       "M8B1"
#define M8_VERSION     1
#define M8_READ_CHUNK  (64 * 1024)
#define M8_NO_SCALE    0xFFFFFFFFu
#define M8_T_Q8        1
#define M8_T_F32       2
#define M8_T_REQUIRED  4
#define M8_TIER_ANY    2
#define M8_SANITY_IDS  8

#define OFF_MAGIC            0
#define OFF_VERSION          4
#define OFF_HEADER_BYTES     6
#define OFF_FILE_BYTES       8
#define OFF_PAYLOAD_BYTES    12
#define OFF_FLAGS            16
#define OFF_HEADER_CRC       20
#define OFF_PAYLOAD_SHA      24
#define OFF_DIM              56
#define OFF_HIDDEN           58
#define OFF_LAYERS           60
#define OFF_HEADS            62
#define OFF_KV_HEADS         64
#define OFF_VOCAB            66
#define OFF_SEQ              68
#define OFF_CTX              70
#define OFF_QUANT_BITS       72
#define OFF_ROW_ALIGN        73
#define OFF_TENSOR_COUNT     74
#define OFF_Q_BYTES          76
#define OFF_SCALE_BYTES      80
#define OFF_NORM_BYTES       84
#define OFF_TOK_OFF          88
#define OFF_TOK_BYTES        92
#define OFF_SANITY_LEN       96
#define OFF_SANITY_TOPK      97
#define OFF_SANITY_CRC       100
#define OFF_SANITY_PROMPT    104
#define OFF_SANITY_IDS       120
#define OFF_NAME             140
#define OFF_BUILD_ID         172

static uint16_t rd16(const uint8_t *p)
{
    return (uint16_t)(p[0] | ((uint16_t)p[1] << 8));
}

static uint32_t rd32(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static void copy_str(char *dst, const uint8_t *src, size_t n)
{
    memcpy(dst, src, n);
    dst[n] = '\0';
}

// CRC over the 4 KiB header with the header_crc32 field treated as zero.
// esp_rom_crc32_le(0, ...) is the standard reflected CRC-32 with init/xorout
// 0xFFFFFFFF (the ROM wrapper inverts at entry and exit), i.e. zlib.crc32.
static uint32_t header_crc(const uint8_t *hdr)
{
    static const uint8_t zero4[4] = { 0, 0, 0, 0 };
    uint32_t crc = esp_rom_crc32_le(0, hdr, OFF_HEADER_CRC);
    crc = esp_rom_crc32_le(crc, zero4, sizeof(zero4));
    crc = esp_rom_crc32_le(crc, hdr + OFF_PAYLOAD_SHA,
                           M8_HEADER_BYTES - OFF_PAYLOAD_SHA);
    return crc;
}

static int sha256_payload(const uint8_t *p, size_t n, uint8_t out[32])
{
    mbedtls_sha256_context ctx;
    mbedtls_sha256_init(&ctx);
    int rc = mbedtls_sha256_starts(&ctx, 0);
    if (rc == 0) {
        rc = mbedtls_sha256_update(&ctx, p, n);
    }
    if (rc == 0) {
        rc = mbedtls_sha256_finish(&ctx, out);
    }
    mbedtls_sha256_free(&ctx);
    return rc;
}

static bool expected_shape(int id, uint32_t rows, uint32_t cols,
                           int dim, int hid, int nl, int kv, int vocab)
{
    switch (id) {
    case 1:  return rows == (uint32_t)vocab && cols == (uint32_t)dim;
    case 2:  return rows == (uint32_t)nl * dim && cols == (uint32_t)dim;
    case 3:  return rows == (uint32_t)nl * kv && cols == (uint32_t)dim;
    case 4:  return rows == (uint32_t)nl * kv && cols == (uint32_t)dim;
    case 5:  return rows == (uint32_t)nl * dim && cols == (uint32_t)dim;
    case 6:  return rows == (uint32_t)nl * hid && cols == (uint32_t)dim;
    case 7:  return rows == (uint32_t)nl * dim && cols == (uint32_t)hid;
    case 8:  return rows == (uint32_t)nl * hid && cols == (uint32_t)dim;
    case 9:  return rows == (uint32_t)nl && cols == (uint32_t)dim;
    case 10: return rows == (uint32_t)nl && cols == (uint32_t)dim;
    case 11: return rows == 1 && cols == (uint32_t)dim;
    default: return false;
    }
}

static int parse_header(model_stage_t *st)
{
    uint8_t *h = st->blob;
    if (st->file_bytes < M8_HEADER_BYTES) {
        return MODEL_ERR_SIZE;
    }
    if (memcmp(h + OFF_MAGIC, M8_MAGIC, 4) != 0) {
        ESP_LOGE(TAG, "bad magic %02x%02x%02x%02x", h[0], h[1], h[2], h[3]);
        return MODEL_ERR_MAGIC;
    }
    if (rd16(h + OFF_VERSION) != M8_VERSION ||
        rd16(h + OFF_HEADER_BYTES) != M8_HEADER_BYTES) {
        ESP_LOGE(TAG, "unsupported version/header %u/%u",
                 rd16(h + OFF_VERSION), rd16(h + OFF_HEADER_BYTES));
        return MODEL_ERR_VERSION;
    }
    uint32_t file_bytes = rd32(h + OFF_FILE_BYTES);
    uint32_t payload_bytes = rd32(h + OFF_PAYLOAD_BYTES);
    if (file_bytes != st->file_bytes ||
        payload_bytes != st->file_bytes - M8_HEADER_BYTES) {
        ESP_LOGE(TAG, "size mismatch header=%u actual=%u payload=%u",
                 (unsigned)file_bytes, (unsigned)st->file_bytes,
                 (unsigned)payload_bytes);
        return MODEL_ERR_SIZE;
    }
    if (!(rd32(h + OFF_FLAGS) & 1u)) {
        ESP_LOGE(TAG, "tokenizer is not embedded");
        return MODEL_ERR_VERSION;
    }
    if (header_crc(h) != rd32(h + OFF_HEADER_CRC)) {
        ESP_LOGE(TAG, "header CRC mismatch");
        return MODEL_ERR_HEADER_CRC;
    }
    uint8_t digest[32];
    if (sha256_payload(st->payload, st->payload_bytes, digest) != 0) {
        ESP_LOGE(TAG, "SHA-256 failed");
        return MODEL_ERR_IO;
    }
    if (memcmp(digest, h + OFF_PAYLOAD_SHA, sizeof(digest)) != 0) {
        ESP_LOGE(TAG, "payload SHA-256 mismatch");
        return MODEL_ERR_SHA;
    }

    st->dim = rd16(h + OFF_DIM);
    st->hidden = rd16(h + OFF_HIDDEN);
    st->layers = rd16(h + OFF_LAYERS);
    st->heads = rd16(h + OFF_HEADS);
    st->kv_heads = rd16(h + OFF_KV_HEADS);
    st->vocab = rd16(h + OFF_VOCAB);
    st->seq = rd16(h + OFF_SEQ);
    st->ctx = rd16(h + OFF_CTX);
    if (st->dim <= 0 || st->hidden <= 0 || st->layers <= 0 ||
        st->heads <= 0 || st->kv_heads <= 0 || st->vocab <= 0 ||
        st->seq <= 0 || st->ctx <= 0 || st->ctx > st->seq ||
        st->heads % st->kv_heads != 0 || st->dim % st->heads != 0) {
        ESP_LOGE(TAG, "invalid config dim=%d hidden=%d layers=%d heads=%d kv=%d "
                      "vocab=%d seq=%d ctx=%d",
                 st->dim, st->hidden, st->layers, st->heads, st->kv_heads,
                 st->vocab, st->seq, st->ctx);
        return MODEL_ERR_SHAPE;
    }
    if (h[OFF_QUANT_BITS] != 8 || h[OFF_ROW_ALIGN] != 16) {
        ESP_LOGE(TAG, "unsupported quant_bits/row_align %u/%u",
                 h[OFF_QUANT_BITS], h[OFF_ROW_ALIGN]);
        return MODEL_ERR_VERSION;
    }

    st->q_bytes = rd32(h + OFF_Q_BYTES);
    st->scale_bytes = rd32(h + OFF_SCALE_BYTES);
    st->norm_bytes = rd32(h + OFF_NORM_BYTES);
    st->tok_off = rd32(h + OFF_TOK_OFF);
    st->tok_bytes = rd32(h + OFF_TOK_BYTES);
    if (st->tok_off % 4 != 0 || st->tok_bytes < 8 ||
        (uint64_t)st->tok_off + st->tok_bytes > st->payload_bytes) {
        ESP_LOGE(TAG, "tokenizer range invalid");
        return MODEL_ERR_SHAPE;
    }

    st->sanity_prompt_len = h[OFF_SANITY_LEN];
    st->sanity_topk = h[OFF_SANITY_TOPK];
    st->sanity_crc = rd32(h + OFF_SANITY_CRC);
    if (st->sanity_prompt_len > 8 || st->sanity_topk > 32 ||
        (st->sanity_prompt_len > 0 && st->sanity_topk == 0)) {
        ESP_LOGE(TAG, "invalid sanity header");
        return MODEL_ERR_SHAPE;
    }
    for (int i = 0; i < 8; i++) {
        st->sanity_prompt[i] = rd16(h + OFF_SANITY_PROMPT + 2 * i);
        st->sanity_ids[i] = rd16(h + OFF_SANITY_IDS + 2 * i);
    }
    copy_str(st->name, h + OFF_NAME, 32);
    copy_str(st->build_id, h + OFF_BUILD_ID, 32);

    int count = rd16(h + OFF_TENSOR_COUNT);
    if (count <= 0 || count > M8_MAX_TENSORS) {
        ESP_LOGE(TAG, "tensor_count %d", count);
        return MODEL_ERR_SHAPE;
    }
    st->tensor_count = count;

    const int kv = st->dim * st->kv_heads / st->heads;
    bool seen[12] = { false };
    uint32_t qsum = 0, ssum = 0, nsum = 0, max_end = 0;
    for (int i = 0; i < count; i++) {
        const uint8_t *e = h + M8_TENSOR_OFF + (size_t)i * M8_TENSOR_BYTES;
        m8_tensor_t *t = &st->tensors[i];
        t->id = rd16(e + 0);
        t->flags = rd16(e + 2);
        t->rows = rd32(e + 4);
        t->cols = rd32(e + 8);
        t->stride = rd32(e + 12);
        t->data_offset = rd32(e + 16);
        t->data_bytes = rd32(e + 20);
        t->scale_offset = rd32(e + 24);
        t->scale_bytes = rd32(e + 28);
        t->tier = e[32];

        if (t->id < 1 || t->id > 11 || seen[t->id]) {
            ESP_LOGE(TAG, "tensor %d: bad/duplicate id %u", i, t->id);
            return MODEL_ERR_SHAPE;
        }
        seen[t->id] = true;
        if (t->tier > M8_TIER_ANY ||
            !expected_shape(t->id, t->rows, t->cols, st->dim, st->hidden,
                            st->layers, kv, st->vocab)) {
            ESP_LOGE(TAG, "tensor id=%u: unexpected shape %ux%u",
                     t->id, (unsigned)t->rows, (unsigned)t->cols);
            return MODEL_ERR_SHAPE;
        }
        int kind = t->flags & (M8_T_Q8 | M8_T_F32);
        if (kind == M8_T_Q8) {
            if (t->id > 8) {
                return MODEL_ERR_SHAPE;
            }
            if (t->stride % 16 != 0 || t->stride < t->cols ||
                t->data_bytes != (uint64_t)t->rows * t->stride ||
                t->data_offset % 16 != 0 ||
                t->scale_offset == M8_NO_SCALE || t->scale_offset % 4 != 0 ||
                t->scale_bytes != (uint64_t)t->rows * 4) {
                ESP_LOGE(TAG, "tensor id=%u: bad q8 layout", t->id);
                return MODEL_ERR_SHAPE;
            }
            if ((uint64_t)t->data_offset + t->data_bytes > st->payload_bytes ||
                (uint64_t)t->scale_offset + t->scale_bytes > st->payload_bytes) {
                ESP_LOGE(TAG, "tensor id=%u: q8 range outside payload", t->id);
                return MODEL_ERR_SHAPE;
            }
            qsum += t->data_bytes;
            ssum += t->scale_bytes;
            if (t->data_offset + t->data_bytes > max_end) {
                max_end = t->data_offset + t->data_bytes;
            }
            if (t->scale_offset + t->scale_bytes > max_end) {
                max_end = t->scale_offset + t->scale_bytes;
            }
        } else if (kind == M8_T_F32) {
            if (t->id < 9) {
                return MODEL_ERR_SHAPE;
            }
            if (t->data_bytes != (uint64_t)t->rows * t->cols * 4 ||
                t->data_offset % 4 != 0 ||
                (uint64_t)t->data_offset + t->data_bytes > st->payload_bytes) {
                ESP_LOGE(TAG, "tensor id=%u: bad f32 layout", t->id);
                return MODEL_ERR_SHAPE;
            }
            nsum += t->data_bytes;
            if (t->data_offset + t->data_bytes > max_end) {
                max_end = t->data_offset + t->data_bytes;
            }
        } else {
            ESP_LOGE(TAG, "tensor id=%u: flags 0x%x", t->id, t->flags);
            return MODEL_ERR_SHAPE;
        }
    }
    for (int id = 1; id <= 11; id++) {
        if (!seen[id]) {
            ESP_LOGE(TAG, "missing tensor id=%d", id);
            return MODEL_ERR_SHAPE;
        }
    }
    if (qsum != st->q_bytes || ssum != st->scale_bytes ||
        nsum != st->norm_bytes) {
        ESP_LOGE(TAG, "byte totals q=%u/%u s=%u/%u n=%u/%u",
                 (unsigned)qsum, (unsigned)st->q_bytes,
                 (unsigned)ssum, (unsigned)st->scale_bytes,
                 (unsigned)nsum, (unsigned)st->norm_bytes);
        return MODEL_ERR_SHAPE;
    }
    if (st->tok_off < max_end) {
        ESP_LOGE(TAG, "tokenizer overlaps tensor data");
        return MODEL_ERR_SHAPE;
    }

    ESP_LOGI(TAG, "blob ok: %u B, name=\"%s\" build=\"%s\"",
             (unsigned)st->file_bytes, st->name, st->build_id);
    ESP_LOGI(TAG, "  cfg dim=%d hidden=%d layers=%d heads=%d kv=%d vocab=%d "
                  "seq=%d ctx=%d",
             st->dim, st->hidden, st->layers, st->heads, st->kv_heads,
             st->vocab, st->seq, st->ctx);
    ESP_LOGI(TAG, "  payload q=%u scales=%u norms=%u tokenizer=%u @ %u",
             (unsigned)st->q_bytes, (unsigned)st->scale_bytes,
             (unsigned)st->norm_bytes, (unsigned)st->tok_bytes,
             (unsigned)st->tok_off);
    ESP_LOGI(TAG, "  sanity prompt=%d top%d crc=0x%08x",
             st->sanity_prompt_len, st->sanity_topk,
             (unsigned)st->sanity_crc);
    return MODEL_OK;
}

const char *model_strerror(int rc)
{
    switch (rc) {
    case MODEL_OK:            return "ok";
    case MODEL_ERR_IO:        return "io";
    case MODEL_ERR_SIZE:      return "size";
    case MODEL_ERR_MAGIC:     return "magic";
    case MODEL_ERR_VERSION:   return "version";
    case MODEL_ERR_HEADER_CRC:return "header-crc";
    case MODEL_ERR_SHA:       return "payload-sha";
    case MODEL_ERR_SHAPE:     return "shape";
    case MODEL_ERR_TOKENIZER: return "tokenizer";
    case MODEL_ERR_NOMEM:     return "nomem";
    case MODEL_ERR_SANITY:    return "sanity";
    default:                  return "?";
    }
}

void model_stage_free(model_stage_t *st)
{
    if (!st) {
        return;
    }
    free(st->blob);
    memset(st, 0, sizeof(*st));
}

int model_stage_file(const char *path, model_stage_t *st)
{
    memset(st, 0, sizeof(*st));
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        ESP_LOGW(TAG, "open %s failed", path);
        return MODEL_ERR_IO;
    }
    struct stat sb;
    if (fstat(fd, &sb) != 0 || sb.st_size < M8_HEADER_BYTES ||
        sb.st_size > M8_MAX_BYTES) {
        ESP_LOGE(TAG, "%s: bad size", path);
        close(fd);
        return MODEL_ERR_SIZE;
    }
    size_t sz = (size_t)sb.st_size;
    uint8_t *blob = heap_caps_malloc(sz, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!blob) {
        ESP_LOGE(TAG, "no PSRAM for %u B blob", (unsigned)sz);
        close(fd);
        return MODEL_ERR_NOMEM;
    }
    size_t off = 0;
    while (off < sz) {
        size_t want = sz - off;
        if (want > M8_READ_CHUNK) {
            want = M8_READ_CHUNK;
        }
        ssize_t n = read(fd, blob + off, want);
        if (n <= 0) {
            ESP_LOGE(TAG, "%s: short read at %u/%u", path,
                     (unsigned)off, (unsigned)sz);
            free(blob);
            close(fd);
            return MODEL_ERR_IO;
        }
        off += (size_t)n;
    }
    close(fd);

    st->blob = blob;
    st->file_bytes = sz;
    st->payload = blob + M8_HEADER_BYTES;
    st->payload_bytes = sz - M8_HEADER_BYTES;
    int rc = parse_header(st);
    if (rc != MODEL_OK) {
        model_stage_free(st);
    }
    return rc;
}

int model_stage_partition(const char *label, model_stage_t *st)
{
    memset(st, 0, sizeof(*st));
    const esp_partition_t *p = esp_partition_find_first(
        ESP_PARTITION_TYPE_DATA, 0x40, label);
    if (!p || p->size < 4) {
        ESP_LOGW(TAG, "partition %s not found", label);
        return MODEL_ERR_IO;
    }
    uint32_t len = 0;
    if (esp_partition_read(p, 0, &len, sizeof(len)) != ESP_OK) {
        return MODEL_ERR_IO;
    }
    if (len < M8_HEADER_BYTES || len > M8_MAX_BYTES || len > p->size - 4) {
        ESP_LOGW(TAG, "partition %s: bad length %u", label, (unsigned)len);
        return MODEL_ERR_SIZE;
    }
    uint8_t *blob = heap_caps_malloc(len, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!blob) {
        return MODEL_ERR_NOMEM;
    }
    if (esp_partition_read(p, 4, blob, len) != ESP_OK) {
        free(blob);
        return MODEL_ERR_IO;
    }
    st->blob = blob;
    st->file_bytes = len;
    st->payload = blob + M8_HEADER_BYTES;
    st->payload_bytes = len - M8_HEADER_BYTES;
    int rc = parse_header(st);
    if (rc != MODEL_OK) {
        model_stage_free(st);
    }
    return rc;
}

int model_stage_save(const model_stage_t *st, const char *path)
{
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        ESP_LOGE(TAG, "create %s failed", path);
        return MODEL_ERR_IO;
    }
    size_t off = 0;
    while (off < st->file_bytes) {
        size_t want = st->file_bytes - off;
        if (want > M8_READ_CHUNK) {
            want = M8_READ_CHUNK;
        }
        ssize_t n = write(fd, st->blob + off, want);
        if (n <= 0) {
            ESP_LOGE(TAG, "short write at %u/%u", (unsigned)off,
                     (unsigned)st->file_bytes);
            close(fd);
            return MODEL_ERR_IO;
        }
        off += (size_t)n;
    }
    close(fd);
    return MODEL_OK;
}

static qmat_t *qmat_for(llm_t *m, int id)
{
    switch (id) {
    case 1:  return &m->tok_emb;
    case 2:  return &m->wq;
    case 3:  return &m->wk;
    case 4:  return &m->wv;
    case 5:  return &m->wo;
    case 6:  return &m->w1;
    case 7:  return &m->w2;
    case 8:  return &m->w3;
    default: return NULL;
    }
}

// Descending top-k with ties broken by the lower index, matching the host
// logit_dump.py:topk() (and the packer's fingerprint).
static void topk_ids(const float *v, int n, int k, int *ids)
{
    int ntop = 0;
    for (int i = 0; i < n; i++) {
        if (ntop < k) {
            int pos = ntop;
            while (pos > 0 && (v[i] > v[ids[pos - 1]] ||
                               (v[i] == v[ids[pos - 1]] && i < ids[pos - 1]))) {
                ids[pos] = ids[pos - 1];
                pos--;
            }
            ids[pos] = i;
            ntop++;
        } else if (v[i] > v[ids[k - 1]]) {
            int pos = k - 1;
            while (pos > 0 && (v[i] > v[ids[pos - 1]] ||
                               (v[i] == v[ids[pos - 1]] && i < ids[pos - 1]))) {
                ids[pos] = ids[pos - 1];
                pos--;
            }
            ids[pos] = i;
        }
    }
}

static uint32_t topk_crc(const float *v, int n, int k, int *ids)
{
    topk_ids(v, n, k, ids);
    uint8_t buf[32 * 4];
    for (int i = 0; i < k; i++) {
        uint32_t bits;
        memcpy(&bits, &v[ids[i]], sizeof(bits));
        buf[4 * i + 0] = (uint8_t)bits;
        buf[4 * i + 1] = (uint8_t)(bits >> 8);
        buf[4 * i + 2] = (uint8_t)(bits >> 16);
        buf[4 * i + 3] = (uint8_t)(bits >> 24);
    }
    return esp_rom_crc32_le(0, buf, (uint32_t)(k * 4));
}

static int m8_sanity(const model_stage_t *st, llm_t *m)
{
    if (st->sanity_prompt_len == 0) {
        ESP_LOGW(TAG, "blob has no sanity fingerprint");
        return MODEL_OK;
    }
    int pos = 0;
    float *lg = m->logits;
    for (int i = 0; i < st->sanity_prompt_len; i++) {
        if (llm_forward_int8(m, st->sanity_prompt[i], pos++, lg) != 0) {
            ESP_LOGE(TAG, "sanity forward failed at prompt token %d", i);
            return MODEL_ERR_SANITY;
        }
    }
    int ids[32];
    uint32_t crc = topk_crc(lg, m->vocab_size, st->sanity_topk, ids);
    if (crc != st->sanity_crc) {
        ESP_LOGE(TAG, "sanity CRC 0x%08x != 0x%08x (top1=%d)",
                 (unsigned)crc, (unsigned)st->sanity_crc, ids[0]);
        return MODEL_ERR_SANITY;
    }
    for (int i = 0; i < M8_SANITY_IDS; i++) {
        int best = 0;
        float bv = lg[0];
        for (int j = 1; j < m->vocab_size; j++) {
            if (lg[j] > bv) {
                bv = lg[j];
                best = j;
            }
        }
        if (best != (int)st->sanity_ids[i]) {
            ESP_LOGE(TAG, "sanity greedy[%d] %d != %d",
                     i, best, (int)st->sanity_ids[i]);
            return MODEL_ERR_SANITY;
        }
        if (i + 1 < M8_SANITY_IDS &&
            llm_forward_int8(m, best, pos++, lg) != 0) {
            return MODEL_ERR_SANITY;
        }
    }
    ESP_LOGI(TAG, "sanity ok: top%d crc=0x%08x greedy=%d,%d,%d,...",
             st->sanity_topk, (unsigned)crc,
             st->sanity_ids[0], st->sanity_ids[1], st->sanity_ids[2]);
    return MODEL_OK;
}

int model_commit(model_stage_t *st, llm_t *m, tokenizer_t *tk)
{
    memset(tk, 0, sizeof(*tk));
    int rc = llm_init_config(m, st->dim, st->hidden, st->layers, st->heads,
                             st->kv_heads, st->vocab, st->seq, st->ctx);
    if (rc != 0) {
        ESP_LOGE(TAG, "llm_init_config rc=%d", rc);
        return MODEL_ERR_NOMEM;
    }
    // No legacy fp32 buffer; the int8 path only reads the RMS norms.
    m->raw = NULL;
    m->raw_size = 0;

    size_t norm_n = (size_t)2 * st->layers * st->dim + st->dim;
    m->rms_store = heap_caps_malloc(norm_n * sizeof(float),
                                    MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!m->rms_store) {
        ESP_LOGE(TAG, "no room for %u B of norms", (unsigned)(norm_n * 4));
        llm_free_state(m);
        return MODEL_ERR_NOMEM;
    }
    m->f_rms_att = m->rms_store;
    m->f_rms_ffn = m->rms_store + (size_t)st->layers * st->dim;
    m->f_rms_final = m->rms_store + (size_t)2 * st->layers * st->dim;

    if (llm_int8_alloc(m, true) != 0) {
        ESP_LOGE(TAG, "int8 arena allocation failed");
        llm_free_state(m);
        return MODEL_ERR_NOMEM;
    }

    for (int i = 0; i < st->tensor_count; i++) {
        const m8_tensor_t *t = &st->tensors[i];
        if (t->flags & M8_T_Q8) {
            qmat_t *q = qmat_for(m, t->id);
            if (!q || q->rows != (int)t->rows || q->cols != (int)t->cols ||
                q->stride != (int)t->stride) {
                ESP_LOGE(TAG, "tensor id=%u does not fit the arena", t->id);
                llm_free_state(m);
                return MODEL_ERR_SHAPE;
            }
            memcpy(q->q, st->payload + t->data_offset, t->data_bytes);
            memcpy(q->scale, st->payload + t->scale_offset, t->scale_bytes);
        } else {
            float *dst;
            if (t->id == 9) {
                dst = m->rms_store;
            } else if (t->id == 10) {
                dst = m->rms_store + (size_t)st->layers * st->dim;
            } else if (t->id == 11) {
                dst = m->rms_store + (size_t)2 * st->layers * st->dim;
            } else {
                llm_free_state(m);
                return MODEL_ERR_SHAPE;
            }
            memcpy(dst, st->payload + t->data_offset, t->data_bytes);
        }
    }
    m->int8_ready = true;
    llm_int8_report(m);

    if (tok_parse(tk, st->payload + st->tok_off, st->tok_bytes, st->vocab) != 0) {
        ESP_LOGE(TAG, "embedded tokenizer rejected");
        tok_free(tk);
        llm_free_state(m);
        return MODEL_ERR_TOKENIZER;
    }
    ESP_LOGI(TAG, "tokenizer: vocab=%d max_token_len=%d",
             tk->vocab_size, tk->max_token_length);

    rc = m8_sanity(st, m);
    if (rc != MODEL_OK) {
        tok_free(tk);
        llm_free_state(m);
        return rc;
    }
    return MODEL_OK;
}

int model_load_m8(const char *path, llm_t *m, tokenizer_t *tk)
{
    model_stage_t st;
    int rc = model_stage_file(path, &st);
    if (rc != MODEL_OK) {
        return rc;
    }
    rc = model_commit(&st, m, tk);
    model_stage_free(&st);
    return rc;
}
