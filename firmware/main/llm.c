// Minimal llama2.c (legacy v0) inference for the ESP32-S3.
//
// Two paths run from the same model buffer:
//   - fp32 reference: same math as run.c, used to prove the port is correct
//   - int8: per-row quantized weights + quantized activations, matmuls go
//     through the 128-bit SIMD dot product in dot_s8.S
//
// The whole model file is held in PSRAM. The int8 copies prefer internal SRAM
// because that is where the SIMD loop actually benefits from low latency.

#include "llm.h"
#include "bench_cfg.h"

#include <math.h>
#include <string.h>

#include "esp_heap_caps.h"
#include "esp_cpu.h"
#include "esp_log.h"
#include "esp_memory_utils.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#if DMA_PREFETCH
#include "esp_async_memcpy.h"
#endif

static const char *TAG = "llm";

static void *al(size_t n, bool internal, bool align16)
{
    uint32_t caps = (internal ? MALLOC_CAP_INTERNAL : MALLOC_CAP_SPIRAM) | MALLOC_CAP_8BIT;
    void *p = align16 ? heap_caps_aligned_alloc(16, n, caps) : heap_caps_malloc(n, caps);
    if (!p && internal) {
        uint32_t alt = MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT;
        p = align16 ? heap_caps_aligned_alloc(16, n, alt) : heap_caps_malloc(n, alt);
    }
    return p;
}

// ---- avoiding libm calls ---------------------------------------------------
// The S3 has a single-precision FPU (add.s/mul.s/madd.s), but expf/sqrtf/powf
// and cosf/sinf are library routines costing hundreds of cycles each. The
// forward pass calls expf about 1500 times per token, which costs more than all
// the vector MACs together. A pair of interpolated tables replaces them with
// two loads and a multiply-add.

#define LUT_N     1024
#define SILU_MIN  (-16.0f)
#define SILU_STEP 32.0f      // 1024 entries over [-16, 16]
#define EXP_STEP  64.0f      // 1024 entries over [-16, 0]

static float s_silu[LUT_N + 1];
static float s_exp[LUT_N + 1];

static void build_luts(void)
{
    for (int i = 0; i <= LUT_N; i++) {
        float x = SILU_MIN + (float)i / SILU_STEP;
        s_silu[i] = x / (1.0f + expf(-x));
    }
    for (int i = 0; i <= LUT_N; i++) {
        s_exp[i] = expf(-(float)i / EXP_STEP);
    }
}

// lrintf is an external libcall on this target and the int8 quantise loops
// call it once per element, which is tens of thousands of calls per token.
// Round half away from zero instead; that differs from lrintf only on exact
// ties, a 1-LSB change the accuracy check covers.
static inline int round_to_int(float x)
{
    return (int)(x + (x >= 0.0f ? 0.5f : -0.5f));
}

// Clamp without branching: this core has no min.s/max.s, but a pair of
// float selects lowers to olt.s + movf.s/movt.s, which is what we want inside
// a hardware loop where a taken branch costs more than the arithmetic.
static inline float clampf(float t, float lo, float hi)
{
    t = t < lo ? lo : t;
    t = t > hi ? hi : t;
    return t;
}

// x * sigmoid(x), the SwiGLU gate. Interpolated, ~1e-3 relative error.
//
// The index is clamped rather than guarded by early returns: the branchy
// version was costing ~42 cycles per element because the two early exits sat
// inside the compiler's hardware loop. Only the x >= 16 select remains a
// branch, and it is a forward branch that is almost never taken.
static inline float silu_fast(float x)
{
    float t = clampf((x - SILU_MIN) * SILU_STEP, 0.0f, (float)LUT_N);
    int i = (int)t;
    float f = t - (float)i;
    float r = s_silu[i] + f * (s_silu[i + 1] - s_silu[i]);
    return x >= 16.0f ? x : r;
}

// exp(x) for x <= 0 (softmax after subtracting the max). Fully branchless.
static inline float exp_fast(float x)
{
    float t = clampf(-x * EXP_STEP, 0.0f, (float)LUT_N);
    int i = (int)t;
    float f = t - (float)i;
    return s_exp[i] + f * (s_exp[i + 1] - s_exp[i]);
}

// One Newton step from the bit-trick seed: ~0.17% error, no libm call.
static inline float rsqrt_fast(float x)
{
    union {
        float f;
        uint32_t u;
    } v;
    v.f = x;
    v.u = 0x5f3759dfu - (v.u >> 1);
    float y = v.f;
    return y * (1.5f - 0.5f * x * y * y);
}

// Same 16-lane MAC loop as dot_s8.S, but inlined so the compiler keeps the
// ACCX clear outside the row loop. Calling the out-of-line kernel once per row
// cost ~25 cycles per row, which is more than the dot itself at dim 64.
// Requires a 16-byte aligned pair and n a multiple of 16.
static inline int32_t dot_s8_i(const int8_t *a, const int8_t *b, int n)
{
    int32_t acc;
    int32_t cnt = n >> 4;
    __asm__ volatile(
        "movi a14, 0\n\t"
        "wur.accx_0 a14\n\t"
        "wur.accx_1 a14\n\t"
        "1:\n\t"
        "ee.vld.128.ip q0, %1, 16\n\t"
        "ee.vld.128.ip q1, %2, 16\n\t"
        "ee.vmulas.s8.accx q0, q1\n\t"
        "addi %0, %0, -1\n\t"
        "bnez %0, 1b\n\t"
        "rur.accx_0 %3\n\t"
        : "+r"(cnt), "+r"(a), "+r"(b), "=r"(acc)
        :
        : "a14", "memory");
    return acc;
}

static bool take(size_t *off, size_t n, size_t size, size_t *out)
{
    if (*off + n * 4 > size) {
        return false;
    }
    *out = *off;
    *off += n * 4;
    return true;
}

static void rmsnorm(float *out, const float *x, const float *w, int n, bool fast)
{
    float ss = 0.0f;
    for (int i = 0; i < n; i++) {
        ss += x[i] * x[i];
    }
    ss /= n;
    float scale = fast ? rsqrt_fast(ss + 1e-5f) : 1.0f / sqrtf(ss + 1e-5f);
    for (int i = 0; i < n; i++) {
        out[i] = w[i] * (x[i] * scale);
    }
}

// Normalise the scores and, for the int8 V path, fold the V scale into the
// positions at or above tsplit in the same pass. Returns the largest absolute
// folded weight so the caller can pick a quantisation scale without a second
// pass over the weights. Positions below tsplit stay plain normalised scores
// because the hot window accumulates them in fp32.
static float softmax_fold(float *v, int n, bool fast, int tsplit,
                          const float *vs)
{
    float m = v[0];
    for (int i = 1; i < n; i++) {
        if (v[i] > m) {
            m = v[i];
        }
    }
    float s = 0.0f;
    for (int i = 0; i < n; i++) {
        v[i] = fast ? exp_fast(v[i] - m) : expf(v[i] - m);
        s += v[i];
    }
    float inv = 1.0f / s;
    float mx = 0.0f;
    int i = 0;
    for (; i < tsplit; i++) {
        v[i] *= inv;
    }
    for (; i < n; i++) {
        float a = (v[i] * inv) * vs[i];
        v[i] = a;
        float aa = fabsf(a);
        if (aa > mx) {
            mx = aa;
        }
    }
    return mx;
}

// One fp32 head dot. Four independent accumulators because a single serial
// chain costs one FPU add latency per element, which dominates at head_size 8.
static inline float dot_f32_head(const float *a, const float *b, int n)
{
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    int i = 0;
    for (; i + 3 < n; i += 4) {
        s0 += a[i] * b[i];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
    }
    float s = (s0 + s1) + (s2 + s3);
    for (; i < n; i++) {
        s += a[i] * b[i];
    }
    return s;
}

static void mm_f(float *out, const float *x, const float *w, int nrows, int ncols)
{
    for (int r = 0; r < nrows; r++) {
        const float *wr = w + (size_t)r * ncols;
        float s = 0.0f;
        for (int c = 0; c < ncols; c++) {
            s += wr[c] * x[c];
        }
        out[r] = s;
    }
}

static void quant_vec(const float *x, int8_t *qx, int n, int stride, float *scale)
{
    // Max |x| by comparing raw IEEE bit patterns as integers: for positive
    // floats the ordering is the same, and it avoids n soft-float compares.
    union {
        float f;
        uint32_t u;
    } v;
    uint32_t mbits = 0;
    for (int i = 0; i < n; i++) {
        v.f = x[i];
        uint32_t b = v.u & 0x7fffffffu;
        if (b > mbits) {
            mbits = b;
        }
    }
    v.u = mbits;
    float mx = v.f;
    float s = mx > 0.0f ? mx / 127.0f : 1.0f;
    *scale = s;
    float inv = 1.0f / s;
    for (int i = 0; i < n; i++) {
        float t = x[i] * inv;
        int q = (int)(t >= 0.0f ? t + 0.5f : t - 0.5f);
        if (q > 127) {
            q = 127;
        }
        if (q < -127) {
            q = -127;
        }
        qx[i] = (int8_t)q;
    }
    for (int i = n; i < stride; i++) {
        qx[i] = 0;
    }
}

// Matmul against an already-quantized activation (m->qx, scale xs).
static void mm_q_pre(llm_t *m, float *out, const qmat_t *W, int row0, int nrows, float xs)
{
    const int8_t *w = W->q + (size_t)row0 * W->stride;
    const float *sc = W->scale + row0;
    if (W->stride == 64) {
        // Activation is exactly four vector registers; keep it resident and
        // let the weights stream through the fused load+MAC form.
        llm_mm_s8_64(m->qx, w, sc, &xs, out, nrows);
        return;
    }
    for (int r = 0; r < nrows; r++) {
        int32_t d = dot_s8_i(w + (size_t)r * W->stride, m->qx, W->stride);
        out[r] = (float)d * xs * sc[r];
    }
}

static void mm_q(llm_t *m, float *out, const float *x, const qmat_t *W, int row0, int nrows)
{
    float xs;
    quant_vec(x, m->qx, W->cols, m->stride_max, &xs);
    mm_q_pre(m, out, W, row0, nrows, xs);
}

static void luts_init_once(void)
{
    static bool luts_ready;
    if (!luts_ready) {
        int64_t t0 = esp_timer_get_time();
        build_luts();
        luts_ready = true;
        ESP_LOGI(TAG, "exp/silu tables built in %d us", (int)(esp_timer_get_time() - t0));
    }
}

// Allocate the per-token scratch, the K/V caches and the rotary table. Split
// out of llm_parse so the .m8 loader can reuse it: the loader knows the
// dimensions from the blob header and does not have a legacy fp32 buffer.
static int llm_alloc_state(llm_t *m)
{
    const int dim = m->dim, hid = m->hidden_dim, nl = m->n_layers;
    const int kv = m->kv_dim, vocab = m->vocab_size;

    m->x = al(dim * sizeof(float), true, false);
    m->xb = al(dim * sizeof(float), true, false);
    m->xb2 = al(dim * sizeof(float), true, false);
    m->qbuf = al(dim * sizeof(float), true, false);
    m->hb = al(hid * sizeof(float), true, false);
    m->hb2 = al(hid * sizeof(float), true, false);
    m->att = al(m->ctx * sizeof(float), true, false);
    m->logits = al(vocab * sizeof(float), true, false);
    m->key_cache = al((size_t)nl * m->ctx * kv * sizeof(float), false, false);
    m->value_cache = al((size_t)nl * m->ctx * kv * sizeof(float), false, false);
#if KV_INT8
    m->key_q = al((size_t)nl * m->ctx * kv, false, true);
    m->value_q = al((size_t)nl * m->ctx * kv, false, true);
    m->key_scale = al((size_t)nl * m->ctx * m->n_kv_heads * sizeof(float), false, false);
    m->value_scale = al((size_t)nl * m->ctx * m->n_kv_heads * sizeof(float), false, false);
    m->ktmp = al((size_t)kv * sizeof(float), true, false);
    m->vtmp = al((size_t)kv * sizeof(float), true, false);
    m->kdeq = al((size_t)m->head_size * sizeof(float), true, false);
    m->vdeq = al((size_t)m->head_size * sizeof(float), true, false);
    m->qpad = al((size_t)m->n_heads * 16, true, true);
    m->qscale = al((size_t)m->n_heads * sizeof(float), true, false);
    m->a8 = al((size_t)m->ctx, true, true);
#if DMA_PREFETCH
    m->vbuf0 = al((size_t)m->head_size * V_TILE, true, true);
    m->vbuf1 = al((size_t)m->head_size * V_TILE, true, true);
#endif
#endif
#if DUAL_CORE
    // Needed on the fp32 path too: the worker writes its scores into att2.
    // a8_2 is only read by the int8 path but the 512 bytes are negligible.
    m->att2 = al((size_t)m->ctx * sizeof(float), true, false);
    m->a8_2 = al((size_t)m->ctx, true, true);
#endif
    // `hot` positions of K/V for all layers. The size is a build-time policy:
    // every position in the window costs 1.25 KB of internal SRAM that the int8
    // weights would otherwise use. hot=0 disables the window entirely.
    m->hot = HOT_POS;
    if (m->hot < 0) {
        m->hot = 0;
    }
    if (m->hot > m->ctx) {
        m->hot = m->ctx;
    }
    if (m->hot > 0) {
        m->khot = al((size_t)nl * m->n_kv_heads * m->hot * m->head_size * sizeof(float), true, false);
        m->vhot = al((size_t)nl * m->n_kv_heads * m->hot * m->head_size * sizeof(float), true, false);
    }
    if (!m->x || !m->xb || !m->xb2 || !m->qbuf || !m->hb || !m->hb2 ||
        !m->att || !m->logits || !m->key_cache || !m->value_cache ||
        (m->hot > 0 && (!m->khot || !m->vhot))) {
        ESP_LOGE(TAG, "state allocation failed");
        return -4;
    }
#if DUAL_CORE
    if (!m->att2 || !m->a8_2) {
        ESP_LOGE(TAG, "dual core allocation failed");
        return -4;
    }
#endif
#if KV_INT8
    if (!m->key_q || !m->value_q || !m->key_scale || !m->value_scale ||
        !m->ktmp || !m->vtmp || !m->kdeq || !m->vdeq ||
        !m->qpad || !m->qscale || !m->a8
#if DUAL_CORE
        || !m->att2 || !m->a8_2
#endif
#if DMA_PREFETCH
        || !m->vbuf0 || !m->vbuf1
#endif
        ) {
        ESP_LOGE(TAG, "int8 kv allocation failed");
        return -4;
    }
#endif
    m->stride_max = (hid + 15) & ~15;
    if (m->stride_max < ((dim + 15) & ~15)) {
        m->stride_max = (dim + 15) & ~15;
    }
    m->qx = al(m->stride_max, true, true);
    if (!m->qx) {
        ESP_LOGE(TAG, "activation buffer allocation failed");
        return -4;
    }

    // Rotary table. One entry per (position, even dim index); positions past
    // ctx are never used because generation stops there. Deliberately in PSRAM:
    // internal SRAM is needed for the int8 weights, and a 64 KB table here was
    // enough to push the large matrices into PSRAM and lose ~5% throughput.
    m->rope_stride = dim / 2;
    size_t rope_n = (size_t)m->ctx * m->rope_stride;
    m->rope_cos = al(rope_n * sizeof(float), false, false);
    m->rope_sin = al(rope_n * sizeof(float), false, false);
    if (!m->rope_cos || !m->rope_sin) {
        ESP_LOGW(TAG, "no room for the rotary table, will compute it inline");
        m->rope_cos = m->rope_sin = NULL;
        m->rope_stride = 0;
    } else {
        int64_t t0 = esp_timer_get_time();
        for (int pos = 0; pos < m->ctx; pos++) {
            for (int j = 0; j < m->rope_stride; j++) {
                int i = j * 2;
                int head_dim = i % m->head_size;
                float freq = 1.0f / powf(10000.0f, (float)head_dim / (float)m->head_size);
                float val = (float)pos * freq;
                m->rope_cos[(size_t)pos * m->rope_stride + j] = cosf(val);
                m->rope_sin[(size_t)pos * m->rope_stride + j] = sinf(val);
            }
        }
        ESP_LOGI(TAG, "rotary table %d x %d built in %d us",
                 m->ctx, m->rope_stride, (int)(esp_timer_get_time() - t0));
    }
    return 0;
}

void llm_free_state(llm_t *m)
{
#if DUAL_CORE
    // The worker blocks on a notification between layers, so deleting it here
    // is safe as long as no forward pass is in flight.
    if (m->attn_task) {
        vTaskDelete((TaskHandle_t)m->attn_task);
        m->attn_task = NULL;
    }
#endif
#if DMA_PREFETCH
    if (m->dma) {
        esp_async_memcpy_uninstall((async_memcpy_handle_t)m->dma);
        m->dma = NULL;
    }
#endif
    if (m->int8_slab) {
        // q and scale point inside the slab; freeing them individually would
        // be a double free.
        free(m->int8_slab);
        m->int8_slab = NULL;
    } else {
        qmat_t *all[] = { &m->tok_emb, &m->wq, &m->wk, &m->wv,
                          &m->wo, &m->w1, &m->w2, &m->w3 };
        for (int i = 0; i < 8; i++) {
            free(all[i]->q);
            free(all[i]->scale);
        }
    }
    free(m->x);
    free(m->xb);
    free(m->xb2);
    free(m->qbuf);
    free(m->hb);
    free(m->hb2);
    free(m->att);
    free(m->logits);
    free(m->key_cache);
    free(m->value_cache);
#if KV_INT8
    free(m->key_q);
    free(m->value_q);
    free(m->key_scale);
    free(m->value_scale);
    free(m->ktmp);
    free(m->vtmp);
    free(m->kdeq);
    free(m->vdeq);
    free(m->qpad);
    free(m->qscale);
    free(m->a8);
#if DMA_PREFETCH
    free(m->vbuf0);
    free(m->vbuf1);
#endif
#endif
#if DUAL_CORE
    free(m->att2);
    free(m->a8_2);
#endif
    free(m->khot);
    free(m->vhot);
    free(m->rope_cos);
    free(m->rope_sin);
    free(m->qx);
    free(m->rms_store);
    memset(m, 0, sizeof(*m));
}

int llm_init_config(llm_t *m, int dim, int hidden_dim, int n_layers,
                    int n_heads, int n_kv_heads, int vocab_size,
                    int seq_len, int ctx)
{
    luts_init_once();
    memset(m, 0, sizeof(*m));
    if (dim <= 0 || hidden_dim <= 0 || n_layers <= 0 || n_heads <= 0 ||
        n_kv_heads <= 0 || vocab_size <= 0 || seq_len <= 0) {
        ESP_LOGE(TAG, "invalid config");
        return -2;
    }
    if (n_heads % n_kv_heads != 0 || dim % n_heads != 0) {
        ESP_LOGE(TAG, "invalid head geometry");
        return -2;
    }
    m->dim = dim;
    m->hidden_dim = hidden_dim;
    m->n_layers = n_layers;
    m->n_heads = n_heads;
    m->n_kv_heads = n_kv_heads;
    m->vocab_size = vocab_size;
    m->seq_len = seq_len;
    m->kv_dim = dim * n_kv_heads / n_heads;
    m->head_size = dim / n_heads;
    m->kv_mul = n_heads / n_kv_heads;
    m->ctx = (ctx > 0 && ctx < seq_len) ? ctx : seq_len;
    int rc = llm_alloc_state(m);
    if (rc != 0) {
        llm_free_state(m);
        return rc;
    }
    return 0;
}

int llm_parse(llm_t *m, uint8_t *raw, size_t size, int ctx)
{
    if (size < 28) {
        memset(m, 0, sizeof(*m));
        return -1;
    }
    int32_t hdr[7];
    memcpy(hdr, raw, sizeof(hdr));
    int rc = llm_init_config(m, hdr[0], hdr[1], hdr[2], hdr[3], hdr[4],
                             hdr[5], hdr[6], ctx);
    if (rc != 0) {
        return rc;
    }
    m->raw = raw;
    m->raw_size = size;

    const int dim = m->dim, hid = m->hidden_dim, nl = m->n_layers;
    const int kv = m->kv_dim, vocab = m->vocab_size;

    size_t off = 28, o;
    if (!take(&off, (size_t)vocab * dim, size, &o)) return -3;
    m->f_tok_emb = (const float *)(raw + o);
    if (!take(&off, (size_t)nl * dim, size, &o)) return -3;
    m->f_rms_att = (const float *)(raw + o);
    if (!take(&off, (size_t)nl * dim * dim, size, &o)) return -3;
    m->f_wq = (const float *)(raw + o);
    if (!take(&off, (size_t)nl * kv * dim, size, &o)) return -3;
    m->f_wk = (const float *)(raw + o);
    if (!take(&off, (size_t)nl * kv * dim, size, &o)) return -3;
    m->f_wv = (const float *)(raw + o);
    if (!take(&off, (size_t)nl * dim * dim, size, &o)) return -3;
    m->f_wo = (const float *)(raw + o);
    if (!take(&off, (size_t)nl * dim, size, &o)) return -3;
    m->f_rms_ffn = (const float *)(raw + o);
    if (!take(&off, (size_t)nl * hid * dim, size, &o)) return -3;
    m->f_w1 = (const float *)(raw + o);
    if (!take(&off, (size_t)nl * dim * hid, size, &o)) return -3;
    m->f_w2 = (const float *)(raw + o);
    if (!take(&off, (size_t)nl * hid * dim, size, &o)) return -3;
    m->f_w3 = (const float *)(raw + o);
    if (!take(&off, (size_t)dim, size, &o)) return -3;
    m->f_rms_final = (const float *)(raw + o);
    // RoPE tables written by the legacy exporter; run.c recomputes them.
    if (!take(&off, (size_t)m->seq_len * m->head_size / 2, size, &o)) return -3;
    if (!take(&off, (size_t)m->seq_len * m->head_size / 2, size, &o)) return -3;
    if (off != size) {
        ESP_LOGW(TAG, "model layout consumed %u of %u bytes", (unsigned)off, (unsigned)size);
    }
    return 0;
}

static int qmat_init(qmat_t *q, int rows, int cols, bool internal)
{
    q->rows = rows;
    q->cols = cols;
    q->stride = (cols + 15) & ~15;
    q->q = al((size_t)rows * q->stride, internal, true);
    q->scale = al((size_t)rows * sizeof(float), internal, false);
    return (q->q && q->scale) ? 0 : -1;
}

static void qmat_fill(qmat_t *q, const float *src)
{
    for (int r = 0; r < q->rows; r++) {
        const float *row = src + (size_t)r * q->cols;
        float mx = 0.0f;
        for (int c = 0; c < q->cols; c++) {
            float a = fabsf(row[c]);
            if (a > mx) {
                mx = a;
            }
        }
        float s = mx > 0.0f ? mx / 127.0f : 1.0f;
        q->scale[r] = s;
        float inv = 1.0f / s;
        int8_t *dst = q->q + (size_t)r * q->stride;
        for (int c = 0; c < q->cols; c++) {
            int v = round_to_int(row[c] * inv);
            if (v > 127) v = 127;
            if (v < -127) v = -127;
            dst[c] = (int8_t)v;
        }
        for (int c = q->cols; c < q->stride; c++) {
            dst[c] = 0;
        }
    }
}

typedef struct {
    qmat_t *q;
    int rows, cols;
    const float *src;
} qspec_t;

static void build_qspecs(llm_t *m, qspec_t s[8])
{
    const int dim = m->dim, hid = m->hidden_dim, nl = m->n_layers;
    const int kv = m->kv_dim, vocab = m->vocab_size;
    s[0] = (qspec_t){ &m->tok_emb, vocab, dim, m->f_tok_emb };
    s[1] = (qspec_t){ &m->wq, nl * dim, dim, m->f_wq };
    s[2] = (qspec_t){ &m->wk, nl * kv, dim, m->f_wk };
    s[3] = (qspec_t){ &m->wv, nl * kv, dim, m->f_wv };
    s[4] = (qspec_t){ &m->wo, nl * dim, dim, m->f_wo };
    s[5] = (qspec_t){ &m->w1, nl * hid, dim, m->f_w1 };
    s[6] = (qspec_t){ &m->w2, nl * dim, hid, m->f_w2 };
    s[7] = (qspec_t){ &m->w3, nl * hid, dim, m->f_w3 };
}

int llm_int8_alloc(llm_t *m, bool prefer_internal)
{
    qspec_t specs[8];
    build_qspecs(m, specs);

    // One slab for everything. Per-matrix allocations fragment the internal
    // heap badly enough that one 55 KB matrix ends up in PSRAM, which costs
    // more than the extra size of a single contiguous block.
    size_t total = 0;
    for (int i = 0; i < 8; i++) {
        specs[i].q->rows = specs[i].rows;
        specs[i].q->cols = specs[i].cols;
        specs[i].q->stride = (specs[i].cols + 15) & ~15;
        total += (size_t)specs[i].rows * specs[i].q->stride;
        total += (size_t)specs[i].rows * sizeof(float);
    }

    uint8_t *slab = prefer_internal ? al(total, true, true) : NULL;
    if (slab && !esp_ptr_internal(slab)) {
        free(slab);
        slab = NULL;
    }
    if (!slab) {
        // Measured on this board: no single free block is big enough for all
        // 274 KB (the largest is ~250 KB even with 345 KB free, because the
        // heap spans several regions). Allocate biggest-first so the small
        // matrices can use the smaller regions instead of one 55 KB matrix
        // being pushed into PSRAM.
        ESP_LOGW(TAG, "no contiguous internal slab for %u B (largest block %u B), "
                      "allocating biggest-first",
                 (unsigned)total,
                 (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL));
    }
    m->int8_slab = slab;

    size_t off = 0;
    int order[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    for (int i = 0; i < 8; i++) {
        for (int j = i + 1; j < 8; j++) {
            size_t a = (size_t)specs[order[i]].rows * specs[order[i]].q->stride;
            size_t b = (size_t)specs[order[j]].rows * specs[order[j]].q->stride;
            if (b > a) {
                int t = order[i];
                order[i] = order[j];
                order[j] = t;
            }
        }
    }
    for (int k = 0; k < 8; k++) {
        int i = order[k];
        size_t n = (size_t)specs[i].rows * specs[i].q->stride;
        if (slab) {
            specs[i].q->q = (int8_t *)(slab + off);
            off += n;
        } else if (qmat_init(specs[i].q, specs[i].rows, specs[i].cols, prefer_internal)) {
            ESP_LOGE(TAG, "int8 allocation failed");
            return -1;
        }
    }
    for (int i = 0; i < 8; i++) {
        size_t n = (size_t)specs[i].rows * sizeof(float);
        if (slab) {
            specs[i].q->scale = (float *)(slab + off);
            off += n;
        } else if (!specs[i].q->scale) {
            specs[i].q->scale = al(n, prefer_internal, false);
            if (!specs[i].q->scale) {
                ESP_LOGE(TAG, "int8 scale allocation failed");
                return -1;
            }
        }
    }
    return 0;
}

void llm_int8_report(const llm_t *m)
{
    size_t in_sram = 0, in_psram = 0;
    const qmat_t *all[] = { &m->tok_emb, &m->wq, &m->wk, &m->wv,
                            &m->wo, &m->w1, &m->w2, &m->w3 };
    for (int i = 0; i < 8; i++) {
        size_t n = (size_t)all[i]->rows * all[i]->stride;
        if (esp_ptr_internal(all[i]->q)) {
            in_sram += n;
        } else {
            in_psram += n;
        }
    }
    ESP_LOGI(TAG, "int8 weights placed: %u B internal SRAM, %u B PSRAM (slab %s)",
             (unsigned)in_sram, (unsigned)in_psram, m->int8_slab ? "yes" : "no");
}

int llm_int8_fill(llm_t *m)
{
    qspec_t specs[8];
    build_qspecs(m, specs);
    for (int i = 0; i < 8; i++) {
        if (!specs[i].q->q || !specs[i].q->scale) {
            ESP_LOGE(TAG, "int8 arena incomplete");
            return -1;
        }
        qmat_fill(specs[i].q, specs[i].src);
    }
    m->int8_ready = true;
    llm_int8_report(m);
    return 0;
}

int llm_quantize(llm_t *m, bool prefer_internal)
{
    int rc = llm_int8_alloc(m, prefer_internal);
    if (rc != 0) {
        return rc;
    }
    return llm_int8_fill(m);
}

size_t llm_int8_bytes(const llm_t *m)
{
    size_t n = 0;
    const qmat_t *all[] = { &m->tok_emb, &m->wq, &m->wk, &m->wv, &m->wo, &m->w1, &m->w2, &m->w3 };
    for (int i = 0; i < 8; i++) {
        n += (size_t)all[i]->rows * all[i]->stride;
        n += (size_t)all[i]->rows * sizeof(float);
    }
    return n;
}

void llm_print_mem(const llm_t *m)
{
    size_t w = llm_int8_bytes(m);
    size_t kv = 2 * (size_t)m->n_layers * m->ctx * m->kv_dim * sizeof(float);
    ESP_LOGI(TAG, "int8 weights %u B, fp32 kv cache %u B, model file %u B",
             (unsigned)w, (unsigned)kv, (unsigned)m->raw_size);
#if KV_INT8
    size_t kv8 = 2 * (size_t)m->n_layers * m->ctx * m->kv_dim;
    size_t kvs = 2 * (size_t)m->n_layers * m->ctx * m->n_kv_heads * sizeof(float);
    ESP_LOGI(TAG, "int8 kv %u B + scales %u B (%u B total)",
             (unsigned)kv8, (unsigned)kvs, (unsigned)(kv8 + kvs));
#endif
}

// Quantise one head of K or V. The scale is stored per head per position, so
// the 8 values in a head share a range and the error stays local.
static void quant_head(const float *src, int8_t *dst, float *scale, int n)
{
    float mx = 0.0f;
    for (int i = 0; i < n; i++) {
        float a = fabsf(src[i]);
        if (a > mx) {
            mx = a;
        }
    }
    float s = mx > 0.0f ? mx / 127.0f : 1.0f;
    *scale = s;
    float inv = 1.0f / s;
    for (int i = 0; i < n; i++) {
        int v = round_to_int(src[i] * inv);
        if (v > 127) v = 127;
        if (v < -127) v = -127;
        dst[i] = (int8_t)v;
    }
}

#if KV_INT8
// Quantise the query per head and place each head's 8 values in the half of
// a 16-byte vector that lines up with its KV head inside the 32-byte K record.
// The other half stays zero, so one 16-lane VMULAS yields that head's score.
static void quant_query(const float *q, int8_t *qpad, float *scale,
                        int n_heads, int kv_mul, int hs)
{
    memset(qpad, 0, (size_t)n_heads * 16);
    for (int h = 0; h < n_heads; h++) {
        const float *src = q + h * hs;
        float mx = 0.0f;
        for (int i = 0; i < hs; i++) {
            float a = fabsf(src[i]);
            if (a > mx) {
                mx = a;
            }
        }
        float s = mx > 0.0f ? mx / 127.0f : 1.0f;
        scale[h] = s;
        float inv = 1.0f / s;
        int g = h / kv_mul;
        int8_t *dst = qpad + (size_t)h * 16 + (g & 1) * 8;
        for (int i = 0; i < hs; i++) {
            int v = round_to_int(src[i] * inv);
            if (v > 127) v = 127;
            if (v < -127) v = -127;
            dst[i] = (int8_t)v;
        }
    }
}
#endif

#if KV_INT8 && DMA_PREFETCH
// V is stored in position tiles: [layer][kv_head][tile][dim][tile_pos].
// A whole tile for one kv head is contiguous, so one DMA transaction brings
// all 8 dimension rows into internal SRAM at once.
static inline size_t v_idx(const llm_t *m, int l, int g, int i, int pos)
{
    const int nt = m->ctx / V_TILE;
    return ((((size_t)l * m->n_kv_heads + g) * nt + pos / V_TILE) *
            m->head_size + i) * V_TILE + pos % V_TILE;
}

static bool v_dma_cb(async_memcpy_handle_t h, async_memcpy_event_t *e, void *arg)
{
    (void)h;
    (void)e;
    *(volatile int *)arg = 1;
    return false;
}

static int dma_init(llm_t *m)
{
    if (m->dma) {
        return 0;
    }
    async_memcpy_config_t cfg = ASYNC_MEMCPY_DEFAULT_CONFIG();
    cfg.backlog = 4;
    if (esp_async_memcpy_install(&cfg, (async_memcpy_handle_t *)&m->dma) != ESP_OK) {
        ESP_LOGE(TAG, "async memcpy install failed");
        m->dma = NULL;
        return -1;
    }
    return 0;
}

// Accumulate one head's V contribution from the tiled int8 cache. Tiles are
// double buffered so the DMA for tile t+1 runs while tile t is dotted.
static void v_accum_tiled(llm_t *m, int l, int g, int tsplit, int pos,
                          const int8_t *a8, float sa, float *xbh)
{
    const int hs = m->head_size;
    const int nt = m->ctx / V_TILE;
    const int ti0 = tsplit / V_TILE;
    const int ti1 = pos / V_TILE;
    const size_t tile_bytes = (size_t)hs * V_TILE;
    int8_t *buf[2] = { m->vbuf0, m->vbuf1 };
    volatile int *done[2] = { &m->vdone0, &m->vdone1 };
    const bool use_dma = m->dma != NULL;
    int cur = 0;

    if (use_dma) {
        *done[0] = 0;
        const int8_t *src = m->value_q +
            (((size_t)(l * m->n_kv_heads + g) * nt + ti0) * hs) * V_TILE;
        if (esp_async_memcpy((async_memcpy_handle_t)m->dma, buf[0], (void *)src,
                             tile_bytes, v_dma_cb, (void *)done[0]) != ESP_OK) {
            m->dma = NULL;
            return v_accum_tiled(m, l, g, tsplit, pos, a8, sa, xbh);
        }
    }

    for (int ti = ti0; ti <= ti1; ti++) {
        int nxt = cur ^ 1;
        if (use_dma && ti + 1 <= ti1) {
            *done[nxt] = 0;
            const int8_t *s2 = m->value_q +
                (((size_t)(l * m->n_kv_heads + g) * nt + (ti + 1)) * hs) * V_TILE;
            esp_async_memcpy((async_memcpy_handle_t)m->dma, buf[nxt], (void *)s2,
                             tile_bytes, v_dma_cb, (void *)done[nxt]);
        }
        int tile_start = ti * V_TILE;
        int start = tile_start > tsplit ? tile_start : tsplit;
        int end = (ti + 1) * V_TILE;
        if (end > pos + 1) {
            end = pos + 1;
        }
        int len = end - start;
        int boff = start - tile_start;
        int aoff = start - tsplit;
        if (use_dma) {
            while (!*done[cur]) {
            }
        }
        for (int i = 0; i < hs; i++) {
            const int8_t *src = use_dma
                ? buf[cur] + (size_t)i * V_TILE + boff
                : m->value_q +
                  (((size_t)(l * m->n_kv_heads + g) * nt + ti) * hs + i) * V_TILE + boff;
            int32_t acc = llm_dot_s8(a8 + aoff, src, len);
            xbh[i] += (float)acc * sa;
        }
        cur = nxt;
    }
}
#endif

// Attention for a contiguous range of heads [h0, h1). The caller supplies the
// softmax scratch and the folded-weight buffer, so two cores can run disjoint
// halves without touching each other's memory. The K/V cache is read-only here.
static void attn_heads(llm_t *m, int l, int pos, bool q8,
                       int h0, int h1, float *att, int8_t *a8,
                       uint64_t *aprof)
{
    const int kv = m->kv_dim, hs = m->head_size;
    const float att_scale = 1.0f / sqrtf((float)hs);
    const int tsplit = (pos + 1 < m->hot) ? pos + 1 : m->hot;
    const size_t loff = (size_t)l * m->ctx * kv;

#define ACYC(idx) do { \
        if (aprof) { uint32_t now_ = esp_cpu_get_cycle_count(); \
                     aprof[idx] += (uint32_t)(now_ - pcyc); pcyc = now_; } \
    } while (0)

    for (int h = h0; h < h1; h++) {
        uint32_t pcyc = esp_cpu_get_cycle_count();
        float *qh = m->qbuf + h * hs;
        const int g = h / m->kv_mul;
        const size_t hoff = (size_t)(l * m->n_kv_heads + g) * m->hot * hs;
        const float *khotb = m->khot ? m->khot + hoff : NULL;
        const float *vhotb = m->vhot ? m->vhot + hoff : NULL;
        const float *kb = m->key_cache
            ? m->key_cache + loff + (size_t)g * hs : NULL;
#if KV_INT8
        // Two KV heads share each 16-byte chunk, so g/2 selects the chunk
        // and g&1 selects the half that quant_query() filled.
        const int8_t *kqb = m->key_q + loff + (size_t)(g / 2) * 16;
        // Scales are [layer][kv_head][pos], so one head walks contiguous
        // floats over positions instead of every fourth float.
        const float *ksb = m->key_scale + ((size_t)l * m->n_kv_heads + g) * m->ctx;
        const int8_t *qp = m->qpad + (size_t)h * 16;
        const float qs = m->qscale[h];
#endif
#if KV_INT8
        if (q8) {
            // Hot window in fp32.
            for (int t = 0; t < tsplit; t++) {
                att[t] = dot_f32_head(qh, khotb + (size_t)t * hs, hs) * att_scale;
            }
            // Int8 positions, four at a time. The zero half of qp cancels
            // the neighbouring KV head in the same 16-byte chunk.
            int t = tsplit;
            for (; t + 3 <= pos; t += 4) {
                int32_t acc[4];
                llm_dot16x4_s8(qp, kqb + (size_t)t * kv, kv, acc);
                for (int j = 0; j < 4; j++) {
                    att[t + j] = (float)acc[j] * qs * ksb[t + j] * att_scale;
                }
            }
            for (; t <= pos; t++) {
                int32_t acc = llm_dot16_s8(qp, kqb + (size_t)t * kv);
                att[t] = (float)acc * qs * ksb[t] * att_scale;
            }
        } else
#endif
        {
            for (int t = 0; t <= pos; t++) {
                const float *kh = (t < tsplit) ? (khotb + (size_t)t * hs)
                                               : (kb + (size_t)t * kv);
                att[t] = dot_f32_head(qh, kh, hs) * att_scale;
            }
        }
        ACYC(0);
#if KV_INT8
        const float *vsb = m->value_scale +
            ((size_t)l * m->n_kv_heads + g) * m->ctx;
#else
        const float *vsb = NULL;
#endif
        // Fold the V scale only on the int8 path; the fp32 path wants plain
        // normalised scores for the whole window.
        int fold_at = q8 ? tsplit : pos + 1;
#if !KV_INT8
        // q8 selects int8 weights, not the int8 KV cache, so it can be true in
        // a build with no int8 V path at all. Never fold there, or the fold
        // loop would dereference vs, which is NULL.
        fold_at = pos + 1;
#endif
        float vmax = softmax_fold(att, pos + 1, q8, fold_at, vsb);
#if !KV_INT8
        (void)vmax;
#endif
        float *xbh = m->xb + h * hs;
        memset(xbh, 0, (size_t)hs * sizeof(float));
        const float *vb = m->value_cache
            ? m->value_cache + loff + (size_t)g * hs : NULL;
        // Hot window first, fp32.
        for (int t = 0; t < tsplit; t++) {
            float a = att[t];
            const float *vh = vhotb + (size_t)t * hs;
            for (int i = 0; i < hs; i++) {
                xbh[i] += a * vh[i];
            }
        }
        ACYC(1);
#if KV_INT8
        if (q8 && pos >= tsplit) {
            // Fold the V scale into the softmax weight, quantise the weights
            // per head, then accumulate each output element with one SIMD dot
            // over positions. The V rows are contiguous in t because the
            // layout is [layer][kv_head][dim][pos].
            int n = pos + 1 - tsplit;
            float sa = vmax > 0.0f ? vmax / 127.0f : 1.0f;
            float inv = 1.0f / sa;
            for (int j = 0; j < n; j++) {
                int v = round_to_int(att[tsplit + j] * inv);
                if (v > 127) v = 127;
                if (v < -127) v = -127;
                a8[j] = (int8_t)v;
            }
            ACYC(2);
#if DMA_PREFETCH
            if (DMA_ACTIVE && !m->dma) {
                dma_init(m);
            }
            v_accum_tiled(m, l, g, tsplit, pos, a8, sa, xbh);
#else
            for (int i = 0; i < hs; i++) {
                const int8_t *row = m->value_q +
                    (((size_t)(l * m->n_kv_heads + g) * hs + i) * m->ctx);
                int32_t acc = llm_dot_s8(a8, row + tsplit, n);
                xbh[i] += (float)acc * sa;
            }
#endif
        } else
#endif
        {
            // fp32 cache, t = tsplit..pos.
            for (int t = tsplit; t <= pos; t++) {
                float a = att[t];
                const float *vh = vb + (size_t)t * kv;
                for (int i = 0; i < hs; i++) {
                    xbh[i] += a * vh[i];
                }
            }
        }
        ACYC(3);
    }
#undef ACYC
}

#if DUAL_CORE
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

// Worker core: wait for a layer, run the upper half of the heads, signal back.
static void attn_worker(void *arg)
{
    llm_t *m = (llm_t *)arg;
    const int half = m->n_heads / 2;
    for (;;) {
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        attn_heads(m, m->attn_l, m->attn_pos, m->attn_q8,
                   half, m->n_heads, m->att2, m->a8_2, m->aprof2);
        xTaskNotifyGive((TaskHandle_t)m->attn_main);
    }
}

// Start the worker and pin the calling task to core 0 so the two halves truly
// run in parallel. Called once from llm_forward_int8 on the first int8 token.
static int dc_start(llm_t *m)
{
    if (m->dc_ready) {
        return 0;
    }
    m->attn_main = (void *)xTaskGetCurrentTaskHandle();
    // app_main is already pinned to CPU0 by Kconfig, so the worker on CPU1 is
    // enough to get real parallelism. The legacy dual-core port has no
    // vTaskCoreAffinitySet.
    BaseType_t ok = xTaskCreatePinnedToCore(attn_worker, "attn1", 4096, m,
                                            1, (TaskHandle_t *)&m->attn_task, 1);
    if (ok != pdPASS) {
        ESP_LOGE(TAG, "attn worker create failed");
        return -1;
    }
    m->dc_ready = true;
    return 0;
}
#endif

static int forward_impl(llm_t *m, int token, int pos, float *logits, bool q8)
{
    const int dim = m->dim, kv = m->kv_dim, hid = m->hidden_dim, hs = m->head_size;

    if (token < 0 || token >= m->vocab_size) {
        return -1;
    }
    if (q8 && !m->int8_ready) {
        return -2;
    }

    if (q8) {
        const int8_t *row = m->tok_emb.q + (size_t)token * m->tok_emb.stride;
        float s = m->tok_emb.scale[token];
        for (int i = 0; i < dim; i++) {
            m->x[i] = (float)row[i] * s;
        }
    } else {
        memcpy(m->x, m->f_tok_emb + (size_t)token * dim, (size_t)dim * sizeof(float));
    }

    // Section timing: only accumulated on the int8 path.
    int64_t pt = esp_timer_get_time();
#define PROF(idx) do { \
        if (q8) { int64_t now_ = esp_timer_get_time(); \
                  m->prof[idx] += (uint64_t)(now_ - pt); pt = now_; } \
    } while (0)

    for (int l = 0; l < m->n_layers; l++) {
        rmsnorm(m->xb, m->x, m->f_rms_att + (size_t)l * dim, dim, q8);

        const int loff = l * m->ctx * kv;
        float *k;
        float *v;
#if KV_INT8
        if (q8) {
            k = m->ktmp;
            v = m->vtmp;
        } else {
            k = m->key_cache + loff + (size_t)pos * kv;
            v = m->value_cache + loff + (size_t)pos * kv;
        }
#else
        k = m->key_cache + loff + (size_t)pos * kv;
        v = m->value_cache + loff + (size_t)pos * kv;
#endif

        if (q8) {
            // One quantization of the attention input, three matmuls.
            float xs;
            quant_vec(m->xb, m->qx, dim, m->stride_max, &xs);
            mm_q_pre(m, m->qbuf, &m->wq, l * dim, dim, xs);
            mm_q_pre(m, k, &m->wk, l * kv, kv, xs);
            mm_q_pre(m, v, &m->wv, l * kv, kv, xs);
        } else {
            mm_f(m->qbuf, m->xb, m->f_wq + (size_t)l * dim * dim, dim, dim);
            mm_f(k, m->xb, m->f_wk + (size_t)l * dim * kv, kv, dim);
            mm_f(v, m->xb, m->f_wv + (size_t)l * dim * kv, kv, dim);
        }
        PROF(0);

        for (int i = 0; i < dim; i += 2) {
            float fcr, fci;
            if (m->rope_stride) {
                size_t idx = (size_t)pos * m->rope_stride + (i >> 1);
                fcr = m->rope_cos[idx];
                fci = m->rope_sin[idx];
            } else {
                int head_dim = i % hs;
                float freq = 1.0f / powf(10000.0f, (float)head_dim / (float)hs);
                float val = (float)pos * freq;
                fcr = cosf(val);
                fci = sinf(val);
            }
            int rotn = i < kv ? 2 : 1;
            for (int r = 0; r < rotn; r++) {
                float *vec = r == 0 ? m->qbuf : k;
                float v0 = vec[i];
                float v1 = vec[i + 1];
                vec[i] = v0 * fcr - v1 * fci;
                vec[i + 1] = v0 * fci + v1 * fcr;
            }
        }
        // Mirror the recent positions into internal SRAM. Written once per
        // position and never modified, so this is just a 32-byte copy per head.
        if (pos < m->hot) {
            for (int g = 0; g < m->n_kv_heads; g++) {
                size_t d = ((size_t)(l * m->n_kv_heads + g) * m->hot + pos) * hs;
                const float *ks = k + g * hs;
                const float *vs = v + g * hs;
                float *kd = m->khot + d;
                float *vd = m->vhot + d;
                for (int i = 0; i < hs; i++) {
                    kd[i] = ks[i];
                    vd[i] = vs[i];
                }
            }
        }
#if KV_INT8
        if (q8 && pos >= m->hot) {
            // Quantise the freshly written K/V for this position. One scale
            // per head per position keeps the error local to 8 values, and the
            // hot window above still holds the exact fp32 copy for t < tsplit.
            // Positions below `hot` are never read from this cache, so skip
            // them and pay nothing at short context.
            int8_t *kq = m->key_q + loff + (size_t)pos * kv;
            float *ksc = m->key_scale + ((size_t)l * m->n_kv_heads) * m->ctx + pos;
            float *vsc = m->value_scale + ((size_t)l * m->n_kv_heads) * m->ctx + pos;
            for (int g = 0; g < m->n_kv_heads; g++) {
                quant_head(k + g * hs, kq + g * hs, ksc + (size_t)g * m->ctx, hs);
                // V is stored transposed, [layer][kv_head][dim][pos], so the
                // read side can accumulate each output element with one SIMD
                // dot over positions.
                int8_t vtmp8[32];
                quant_head(v + g * hs, vtmp8, vsc + (size_t)g * m->ctx, hs);
                for (int i = 0; i < hs; i++) {
#if DMA_PREFETCH
                    m->value_q[v_idx(m, l, g, i, pos)] = vtmp8[i];
#else
                    size_t idx = (((size_t)(l * m->n_kv_heads + g) * hs + i)
                                  * m->ctx) + pos;
                    m->value_q[idx] = vtmp8[i];
#endif
                }
            }
        }
#endif
        PROF(1);

#if KV_INT8
        if (q8 && pos >= m->hot) {
            // One quantisation of the query per position. The integer dot
            // below consumes this instead of dequantising K to fp32. Below
            // `hot` every position is served from the fp32 mirror, so skip it.
            quant_query(m->qbuf, m->qpad, m->qscale, m->n_heads, m->kv_mul, hs);
        }
#endif

#if DUAL_CORE
        // Upper half of the heads on the worker core, lower half here. One
        // notification pair per layer; the worker reads the K/V written above.
        if (dc_start(m) == 0) {
            m->attn_l = l;
            m->attn_pos = pos;
            m->attn_q8 = q8;
            xTaskNotifyGive((TaskHandle_t)m->attn_task);
            attn_heads(m, l, pos, q8, 0, m->n_heads / 2, m->att, m->a8, m->aprof);
            ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        } else {
            attn_heads(m, l, pos, q8, 0, m->n_heads, m->att, m->a8, m->aprof);
        }
#else
        attn_heads(m, l, pos, q8, 0, m->n_heads, m->att, m->a8, m->aprof);
#endif
        PROF(2);

        if (q8) {
            mm_q(m, m->xb2, m->xb, &m->wo, l * dim, dim);
        } else {
            mm_f(m->xb2, m->xb, m->f_wo + (size_t)l * dim * dim, dim, dim);
        }
        for (int i = 0; i < dim; i++) {
            m->x[i] += m->xb2[i];
        }
        PROF(3);

        rmsnorm(m->xb, m->x, m->f_rms_ffn + (size_t)l * dim, dim, q8);
        if (q8 && m->moe_hook) {
            // MoE prototype: route this token's FFN and, if the selected
            // expert changed, refill this layer's w1/w2/w3 slice in place.
            m->moe_hook(m, l, m->xb);
        }
        if (q8) {
            float xs;
            quant_vec(m->xb, m->qx, dim, m->stride_max, &xs);
            mm_q_pre(m, m->hb, &m->w1, l * hid, hid, xs);
            mm_q_pre(m, m->hb2, &m->w3, l * hid, hid, xs);
        } else {
            mm_f(m->hb, m->xb, m->f_w1 + (size_t)l * dim * hid, hid, dim);
            mm_f(m->hb2, m->xb, m->f_w3 + (size_t)l * dim * hid, hid, dim);
        }
        PROF(4);
        if (q8) {
            for (int i = 0; i < hid; i++) {
                m->hb[i] = silu_fast(m->hb[i]) * m->hb2[i];
            }
        } else {
            for (int i = 0; i < hid; i++) {
                float val = m->hb[i];
                float sig = 1.0f / (1.0f + expf(-val));
                m->hb[i] = val * sig * m->hb2[i];
            }
        }
        PROF(5);
        if (q8) {
            mm_q(m, m->xb2, m->hb, &m->w2, l * dim, dim);
        } else {
            mm_f(m->xb2, m->hb, m->f_w2 + (size_t)l * hid * dim, dim, hid);
        }
        for (int i = 0; i < dim; i++) {
            m->x[i] += m->xb2[i];
        }
        PROF(6);
    }

    rmsnorm(m->xb, m->x, m->f_rms_final, dim, q8);
    if (q8) {
        mm_q(m, logits, m->xb, &m->tok_emb, 0, m->vocab_size);
    } else {
        mm_f(logits, m->xb, m->f_tok_emb, m->vocab_size, dim);
    }
    PROF(7);
#undef PROF
    return 0;
}

int llm_forward_f32(llm_t *m, int token, int pos, float *logits)
{
    return forward_impl(m, token, pos, logits, false);
}

int llm_forward_int8(llm_t *m, int token, int pos, float *logits)
{
    return forward_impl(m, token, pos, logits, true);
}

void llm_drop_fp32_cache(llm_t *m)
{
#if KV_INT8
    if (m->key_cache) {
        free(m->key_cache);
        m->key_cache = NULL;
    }
    if (m->value_cache) {
        free(m->value_cache);
        m->value_cache = NULL;
    }
#else
    (void)m;
#endif
}
