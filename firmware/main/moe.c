// Synthetic MoE prototype: FFN-only experts swapped into the int8 arena.
//
// The sidecar format and the intent are described in moe.h. This file only
// touches the FFN row ranges of qmat_t; attention, embedding and norms stay
// with the active .m8 model. Nothing here allocates inside a forward pass: the
// sidecar is staged once in PSRAM and moe_bind() attaches it to llm_t.

#include "moe.h"

#include <fcntl.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_partition.h"
#include "esp_rom_crc.h"
#include "esp_timer.h"
#include "mbedtls/sha256.h"

static const char *TAG = "moe";

#define MOE_READ_CHUNK (64 * 1024)

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
#define OFF_EXPERTS          62
#define OFF_ROW_ALIGN        64
#define OFF_MODEL_CRC        68
#define OFF_ROUTER_OFF       72
#define OFF_ROUTER_BYTES     76
#define OFF_EXPERT_BYTES     80
#define OFF_EXPERT_OFF       84
#define OFF_W1Q              88
#define OFF_W1S              96
#define OFF_W2Q              104
#define OFF_W2S              112
#define OFF_W3Q              120
#define OFF_W3S              128
#define OFF_NAME             136
#define OFF_BUILD            168

#define MOE_FLAG_ROUTER_F32  1u
#define MOE_FLAG_SYNTHETIC   2u

// Per-layer expert state plus lifetime counters. Small enough for internal
// SRAM; the sidecar payload itself stays in PSRAM.
typedef struct {
    moe_stage_t st;
    int force;                      // -1 router, 0..E-1 pinned
    bool round_robin;
    uint32_t rr_count[MOE_MAX_LAYERS];
    int cur[MOE_MAX_LAYERS];        // expert currently in the arena, -1 unknown
    moe_stats_t stats;
} moe_ctx_t;

static uint16_t rd16(const uint8_t *p)
{
    return (uint16_t)(p[0] | ((uint16_t)p[1] << 8));
}

static uint32_t rd32(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint32_t align_up(uint32_t n, uint32_t a)
{
    return (n + a - 1) & ~(a - 1);
}

static void copy_str(char *dst, size_t cap, const uint8_t *src, size_t n)
{
    if (n > cap - 1) {
        n = cap - 1;
    }
    memcpy(dst, src, n);
    dst[n] = '\0';
}

static uint32_t header_crc(const uint8_t *hdr)
{
    static const uint8_t zero4[4] = { 0, 0, 0, 0 };
    uint32_t crc = esp_rom_crc32_le(0, hdr, OFF_HEADER_CRC);
    crc = esp_rom_crc32_le(crc, zero4, sizeof(zero4));
    crc = esp_rom_crc32_le(crc, hdr + OFF_PAYLOAD_SHA,
                           MOE_HEADER_BYTES - OFF_PAYLOAD_SHA);
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

// Validate the header and every derived range. `st->file_bytes` and `blob`
// must already be set. When `verify_sha` is true the payload is hashed too.
static int parse_sidecar(moe_stage_t *st, bool verify_sha, bool log)
{
    uint8_t *h = st->blob;
    if (st->file_bytes < MOE_HEADER_BYTES || st->file_bytes > MOE_MAX_BYTES) {
        return MOE_ERR_SIZE;
    }
    if (memcmp(h + OFF_MAGIC, MOE_MAGIC, 4) != 0) {
        ESP_LOGE(TAG, "bad magic %02x%02x%02x%02x", h[0], h[1], h[2], h[3]);
        return MOE_ERR_MAGIC;
    }
    if (rd16(h + OFF_VERSION) != MOE_VERSION ||
        rd16(h + OFF_HEADER_BYTES) != MOE_HEADER_BYTES) {
        ESP_LOGE(TAG, "unsupported version/header %u/%u",
                 rd16(h + OFF_VERSION), rd16(h + OFF_HEADER_BYTES));
        return MOE_ERR_VERSION;
    }
    uint32_t file_bytes = rd32(h + OFF_FILE_BYTES);
    uint32_t payload_bytes = rd32(h + OFF_PAYLOAD_BYTES);
    if (file_bytes != st->file_bytes ||
        payload_bytes != st->file_bytes - MOE_HEADER_BYTES) {
        ESP_LOGE(TAG, "size mismatch header=%u actual=%u payload=%u",
                 (unsigned)file_bytes, (unsigned)st->file_bytes,
                 (unsigned)payload_bytes);
        return MOE_ERR_SIZE;
    }
    uint32_t flags = rd32(h + OFF_FLAGS);
    if ((flags & MOE_FLAG_ROUTER_F32) == 0) {
        ESP_LOGE(TAG, "router is not float32");
        return MOE_ERR_VERSION;
    }
    if (rd32(h + OFF_HEADER_CRC) != header_crc(h)) {
        ESP_LOGE(TAG, "header CRC32 mismatch");
        return MOE_ERR_HEADER_CRC;
    }

    uint16_t dim = rd16(h + OFF_DIM);
    uint16_t hidden = rd16(h + OFF_HIDDEN);
    uint16_t layers = rd16(h + OFF_LAYERS);
    uint16_t experts = rd16(h + OFF_EXPERTS);
    if (dim == 0 || hidden == 0 || layers == 0 || layers > MOE_MAX_LAYERS ||
        experts == 0 || experts > MOE_MAX_EXPERTS) {
        ESP_LOGE(TAG, "bad geometry dim=%u hidden=%u layers=%u experts=%u",
                 dim, hidden, layers, experts);
        return MOE_ERR_SHAPE;
    }
    if (h[OFF_ROW_ALIGN] != 16) {
        ESP_LOGE(TAG, "row_align %u != 16", h[OFF_ROW_ALIGN]);
        return MOE_ERR_SHAPE;
    }

    st->dim = dim;
    st->hidden = hidden;
    st->layers = layers;
    st->experts = experts;
    st->model_sanity_crc = rd32(h + OFF_MODEL_CRC);
    st->router_off = rd32(h + OFF_ROUTER_OFF);
    st->router_bytes = rd32(h + OFF_ROUTER_BYTES);
    st->expert_bytes = rd32(h + OFF_EXPERT_BYTES);
    st->expert_off = rd32(h + OFF_EXPERT_OFF);
    st->w1q_off = rd32(h + OFF_W1Q + 0);
    st->w1q_bytes = rd32(h + OFF_W1Q + 4);
    st->w1s_off = rd32(h + OFF_W1S + 0);
    st->w1s_bytes = rd32(h + OFF_W1S + 4);
    st->w2q_off = rd32(h + OFF_W2Q + 0);
    st->w2q_bytes = rd32(h + OFF_W2Q + 4);
    st->w2s_off = rd32(h + OFF_W2S + 0);
    st->w2s_bytes = rd32(h + OFF_W2S + 4);
    st->w3q_off = rd32(h + OFF_W3Q + 0);
    st->w3q_bytes = rd32(h + OFF_W3Q + 4);
    st->w3s_off = rd32(h + OFF_W3S + 0);
    st->w3s_bytes = rd32(h + OFF_W3S + 4);
    copy_str(st->name, sizeof(st->name), h + OFF_NAME, 32);
    copy_str(st->build, sizeof(st->build), h + OFF_BUILD, 32);

    // Expected tensor sizes for the declared geometry.
    uint32_t stride1 = align_up(dim, 16);
    uint32_t stride2 = align_up(hidden, 16);
    uint64_t rows1 = (uint64_t)layers * hidden;
    uint64_t rows2 = (uint64_t)layers * dim;
    if (st->w1q_bytes != rows1 * stride1 || st->w3q_bytes != rows1 * stride1 ||
        st->w2q_bytes != rows2 * stride2 ||
        st->w1s_bytes != rows1 * 4 || st->w3s_bytes != rows1 * 4 ||
        st->w2s_bytes != rows2 * 4) {
        ESP_LOGE(TAG, "tensor sizes do not match %ux%ux%u",
                 (unsigned)dim, (unsigned)hidden, (unsigned)layers);
        return MOE_ERR_SHAPE;
    }
    uint32_t qoff[3] = { st->w1q_off, st->w2q_off, st->w3q_off };
    uint32_t qlen[3] = { st->w1q_bytes, st->w2q_bytes, st->w3q_bytes };
    uint32_t soff[3] = { st->w1s_off, st->w2s_off, st->w3s_off };
    uint32_t slen[3] = { st->w1s_bytes, st->w2s_bytes, st->w3s_bytes };
    for (int i = 0; i < 3; i++) {
        if (qoff[i] % 16 != 0 || soff[i] % 4 != 0) {
            ESP_LOGE(TAG, "tensor %d misaligned", i);
            return MOE_ERR_SHAPE;
        }
        if ((uint64_t)qoff[i] + qlen[i] > st->expert_bytes ||
            (uint64_t)soff[i] + slen[i] > st->expert_bytes) {
            ESP_LOGE(TAG, "tensor %d outside the expert block", i);
            return MOE_ERR_SHAPE;
        }
    }
    if (st->expert_off % 16 != 0 ||
        (uint64_t)st->expert_off +
            (uint64_t)experts * st->expert_bytes > payload_bytes) {
        ESP_LOGE(TAG, "expert table outside the payload");
        return MOE_ERR_SHAPE;
    }
    if (st->router_off % 4 != 0 ||
        st->router_bytes != (uint32_t)layers * experts * dim * 4 ||
        (uint64_t)st->router_off + st->router_bytes > payload_bytes) {
        ESP_LOGE(TAG, "router table outside the payload");
        return MOE_ERR_SHAPE;
    }

    if (verify_sha) {
        uint8_t digest[32];
        if (sha256_payload(st->blob + MOE_HEADER_BYTES, payload_bytes,
                           digest) != 0) {
            ESP_LOGE(TAG, "SHA-256 failed");
            return MOE_ERR_IO;
        }
        if (memcmp(digest, h + OFF_PAYLOAD_SHA, sizeof(digest)) != 0) {
            ESP_LOGE(TAG, "payload SHA-256 mismatch");
            return MOE_ERR_SHA;
        }
    }

    if (log) {
        ESP_LOGI(TAG, "sidecar ok: %u B, name=\"%s\" build=\"%s\"",
                 (unsigned)st->file_bytes, st->name, st->build);
        ESP_LOGI(TAG, "  cfg dim=%u hidden=%u layers=%u experts=%u "
                      "model_crc=0x%08x",
                 dim, hidden, layers, experts, (unsigned)st->model_sanity_crc);
        ESP_LOGI(TAG, "  router %u B, expert %u B x %u, w1q=%u w2q=%u w3q=%u",
                 (unsigned)st->router_bytes, (unsigned)st->expert_bytes,
                 experts, (unsigned)st->w1q_bytes, (unsigned)st->w2q_bytes,
                 (unsigned)st->w3q_bytes);
    }
    return MOE_OK;
}

int moe_stage_file(const char *path, moe_stage_t *st)
{
    memset(st, 0, sizeof(*st));
    snprintf(st->path, sizeof(st->path), "%s", path);
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        ESP_LOGE(TAG, "open %s failed", path);
        return MOE_ERR_IO;
    }
    struct stat sb;
    if (fstat(fd, &sb) != 0 || sb.st_size < MOE_HEADER_BYTES ||
        sb.st_size > MOE_MAX_BYTES) {
        close(fd);
        return MOE_ERR_SIZE;
    }
    size_t sz = (size_t)sb.st_size;
    uint8_t *blob = heap_caps_malloc(sz, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!blob) {
        close(fd);
        return MOE_ERR_NOMEM;
    }
    size_t got = 0;
    while (got < sz) {
        ssize_t n = read(fd, blob + got, sz - got);
        if (n <= 0) {
            close(fd);
            free(blob);
            return MOE_ERR_IO;
        }
        got += (size_t)n;
    }
    close(fd);

    st->blob = blob;
    st->file_bytes = sz;
    int rc = parse_sidecar(st, true, true);
    if (rc != MOE_OK) {
        moe_stage_free(st);
    }
    return rc;
}

int moe_stage_partition(const char *label, moe_stage_t *st)
{
    memset(st, 0, sizeof(*st));
    snprintf(st->path, sizeof(st->path), "partition:%s", label);
    const esp_partition_t *p = esp_partition_find_first(
        ESP_PARTITION_TYPE_DATA, 0x40, label);
    if (!p || p->size < 4) {
        ESP_LOGW(TAG, "partition %s not found", label);
        return MOE_ERR_IO;
    }
    uint32_t len = 0;
    if (esp_partition_read(p, 0, &len, sizeof(len)) != ESP_OK) {
        return MOE_ERR_IO;
    }
    if (len < MOE_HEADER_BYTES || len > MOE_MAX_BYTES || len > p->size - 4) {
        ESP_LOGW(TAG, "partition %s: bad length %u", label, (unsigned)len);
        return MOE_ERR_SIZE;
    }
    uint8_t *blob = heap_caps_malloc(len, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!blob) {
        return MOE_ERR_NOMEM;
    }
    if (esp_partition_read(p, 4, blob, len) != ESP_OK) {
        free(blob);
        return MOE_ERR_IO;
    }
    st->blob = blob;
    st->file_bytes = len;
    int rc = parse_sidecar(st, true, true);
    if (rc != MOE_OK) {
        moe_stage_free(st);
    }
    return rc;
}

int moe_stage_save(const moe_stage_t *st, const char *path)
{
    if (!st || !st->blob) {
        return MOE_ERR_UNBOUND;
    }
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        ESP_LOGE(TAG, "create %s failed", path);
        return MOE_ERR_IO;
    }
    size_t off = 0;
    while (off < st->file_bytes) {
        size_t want = st->file_bytes - off;
        if (want > MOE_READ_CHUNK) {
            want = MOE_READ_CHUNK;
        }
        ssize_t n = write(fd, st->blob + off, want);
        if (n <= 0) {
            ESP_LOGE(TAG, "short write at %u/%u", (unsigned)off,
                     (unsigned)st->file_bytes);
            close(fd);
            return MOE_ERR_IO;
        }
        off += (size_t)n;
    }
    close(fd);
    return MOE_OK;
}

void moe_stage_free(moe_stage_t *st)
{
    if (!st) {
        return;
    }
    free(st->blob);
    memset(st, 0, sizeof(*st));
}

// Per-layer refill. Runs inside llm_forward_int8(), so it must not allocate,
// log, or take a lock. Counter timing uses esp_timer_get_time(), which is a
// register read on this chip.
static void moe_hook(llm_t *m, int layer, const float *x)
{
    moe_ctx_t *c = (moe_ctx_t *)m->moe;
    if (!c || layer < 0 || layer >= c->st.layers) {
        return;
    }
    const int experts = c->st.experts;
    int e;
    if (c->force >= 0) {
        e = c->force;
    } else if (c->round_robin) {
        e = (int)(c->rr_count[layer]++ % (uint32_t)experts);
    } else {
        const float *r = (const float *)(c->st.blob + MOE_HEADER_BYTES +
                                         c->st.router_off) +
                         (size_t)layer * experts * c->st.dim;
        float best = -FLT_MAX;
        e = 0;
        for (int i = 0; i < experts; i++) {
            const float *ri = r + (size_t)i * c->st.dim;
            float s = 0.0f;
            for (int d = 0; d < c->st.dim; d++) {
                s += ri[d] * x[d];
            }
            if (s > best) {
                best = s;
                e = i;
            }
        }
    }
    c->stats.route_calls++;
    if (e >= 0 && e < experts) {
        c->stats.route_count[e]++;
    }
    c->stats.last_expert[layer] = e;
    if (c->cur[layer] == e) {
        return;
    }

    const uint8_t *blk = c->st.blob + MOE_HEADER_BYTES + c->st.expert_off +
                         (size_t)e * c->st.expert_bytes;
    const int hid = c->st.hidden;
    const int dim = c->st.dim;
    const size_t s1 = (size_t)m->w1.stride;
    const size_t s2 = (size_t)m->w2.stride;
    const size_t r1 = (size_t)layer * hid;
    const size_t r2 = (size_t)layer * dim;

    int64_t t0 = esp_timer_get_time();
    memcpy(m->w1.q + r1 * s1, blk + c->st.w1q_off + r1 * s1,
           (size_t)hid * s1);
    memcpy(m->w3.q + r1 * s1, blk + c->st.w3q_off + r1 * s1,
           (size_t)hid * s1);
    memcpy(m->w2.q + r2 * s2, blk + c->st.w2q_off + r2 * s2,
           (size_t)dim * s2);
    memcpy(m->w1.scale + r1, blk + c->st.w1s_off + r1 * 4,
           (size_t)hid * 4);
    memcpy(m->w3.scale + r1, blk + c->st.w3s_off + r1 * 4,
           (size_t)hid * 4);
    memcpy(m->w2.scale + r2, blk + c->st.w2s_off + r2 * 4,
           (size_t)dim * 4);
    c->stats.copy_us += (uint64_t)(esp_timer_get_time() - t0);
    c->stats.copies++;
    c->stats.copy_bytes +=
        (uint32_t)(2 * hid * s1 + dim * s2 + (2 * hid + dim) * 4);
    c->cur[layer] = e;
}

int moe_bind(llm_t *m, moe_stage_t *st)
{
    if (!m || !st || !st->blob) {
        if (st) {
            moe_stage_free(st);
        }
        return MOE_ERR_UNBOUND;
    }
    if (!m->int8_ready || m->moe) {
        moe_stage_free(st);
        return MOE_ERR_UNBOUND;
    }
    const int dim = st->dim, hid = st->hidden, layers = st->layers;
    const int experts = st->experts;
    if (m->dim != dim || m->hidden_dim != hid || m->n_layers != layers) {
        ESP_LOGE(TAG, "sidecar %dx%dx%d does not match model %dx%dx%d",
                 dim, hid, layers, m->dim, m->hidden_dim, m->n_layers);
        moe_stage_free(st);
        return MOE_ERR_SHAPE;
    }
    const int stride1 = (dim + 15) & ~15;
    const int stride2 = (hid + 15) & ~15;
    if (m->w1.rows != layers * hid || m->w1.cols != dim ||
        m->w1.stride != stride1 || m->w3.rows != layers * hid ||
        m->w3.cols != dim || m->w3.stride != stride1 ||
        m->w2.rows != layers * dim || m->w2.cols != hid ||
        m->w2.stride != stride2 ||
        !m->w1.q || !m->w1.scale || !m->w2.q || !m->w2.scale ||
        !m->w3.q || !m->w3.scale) {
        ESP_LOGE(TAG, "arena layout does not match the sidecar");
        moe_stage_free(st);
        return MOE_ERR_SHAPE;
    }

    moe_ctx_t *c = heap_caps_calloc(1, sizeof(*c),
                                    MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!c) {
        moe_stage_free(st);
        return MOE_ERR_NOMEM;
    }
    c->st = *st;
    memset(st, 0, sizeof(*st));
    c->force = -1;
    for (int i = 0; i < layers; i++) {
        c->cur[i] = -1;
    }
    c->stats.experts = experts;
    c->stats.layers = layers;
    m->moe = c;
    m->moe_hook = moe_hook;
    ESP_LOGI(TAG, "bound %d experts to %d layers; arena w1 %dx%d/%d "
                  "w2 %dx%d/%d w3 %dx%d/%d",
             experts, layers, m->w1.rows, m->w1.cols, m->w1.stride,
             m->w2.rows, m->w2.cols, m->w2.stride,
             m->w3.rows, m->w3.cols, m->w3.stride);
    return MOE_OK;
}

void moe_unbind(llm_t *m)
{
    if (!m) {
        return;
    }
    moe_ctx_t *c = (moe_ctx_t *)m->moe;
    m->moe_hook = NULL;
    m->moe = NULL;
    if (!c) {
        return;
    }
    moe_stage_free(&c->st);
    free(c);
}

void moe_force_expert(llm_t *m, int expert)
{
    moe_ctx_t *c = m ? (moe_ctx_t *)m->moe : NULL;
    if (!c || expert < -1 || expert >= c->st.experts) {
        return;
    }
    c->force = expert;
    if (expert >= 0) {
        c->round_robin = false;
    }
}

void moe_set_round_robin(llm_t *m, bool on)
{
    moe_ctx_t *c = m ? (moe_ctx_t *)m->moe : NULL;
    if (!c) {
        return;
    }
    c->round_robin = on;
    if (on) {
        c->force = -1;
    }
}

void moe_get_stats(const llm_t *m, moe_stats_t *out)
{
    const moe_ctx_t *c = m ? (const moe_ctx_t *)m->moe : NULL;
    if (!out) {
        return;
    }
    if (!c) {
        memset(out, 0, sizeof(*out));
        return;
    }
    *out = c->stats;
}

void moe_reset_stats(llm_t *m)
{
    moe_ctx_t *c = m ? (moe_ctx_t *)m->moe : NULL;
    if (!c) {
        return;
    }
    memset(&c->stats, 0, sizeof(c->stats));
    c->stats.experts = c->st.experts;
    c->stats.layers = c->st.layers;
}

const char *moe_strerror(int rc)
{
    switch (rc) {
    case MOE_OK:            return "ok";
    case MOE_ERR_IO:        return "io";
    case MOE_ERR_SIZE:      return "size";
    case MOE_ERR_MAGIC:     return "magic";
    case MOE_ERR_VERSION:   return "version";
    case MOE_ERR_HEADER_CRC:return "header-crc";
    case MOE_ERR_SHA:       return "sha256";
    case MOE_ERR_SHAPE:     return "shape";
    case MOE_ERR_NOMEM:     return "nomem";
    case MOE_ERR_UNBOUND:   return "unbound";
    default:                return "unknown";
    }
}
