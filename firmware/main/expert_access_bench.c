// PSRAM-resident expert access benchmark — v1.4.0 engineering investigation.
//
// Standalone by design: no model, no MOE1 sidecar, no moe_hook(). It fabricates
// expert blocks with the geometry of the d=96 / hidden=264 design and runs the
// same int8 matmul over them three ways, so the only variable is where the
// weight bytes come from.
//
//   A  per-layer staging   PSRAM -> 79,296 B SRAM slice -> compute
//                          (what the v1.3.0 moe_hook() refill does)
//   B  direct PSRAM        compute from a 16-byte aligned PSRAM pointer
//   C  tiled streaming     PSRAM -> 1/2/4/8/16 KB SRAM tile -> compute
//
// All three move 8 x 79,296 B out of PSRAM per expert forward, so the MB/s
// column is directly comparable. Mode C also has a software-prefetch variant
// that touches the next tile's cache lines before computing the current one.
// That is an ordinary CPU read, not the hardware manual D-cache preload.
//
// Two interpretation caveats:
//   - Mode C stages weight tiles but still reads the row-scale vector straight
//     from PSRAM, so A vs C is not a pure copy-versus-stream comparison.
//   - The "alternating" pattern alternates the expert between layers of one
//     forward pass. It does not yet model a router changing its choice between
//     consecutive tokens at the same layer.
//
// Output is key=value lines on stdout, parsed by
// tools/analyze_expert_access.py.

#include "expert_access_bench.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "esp_cpu.h"
#include "esp_heap_caps.h"
#include "llm.h"

#define EA_DIM        96
#define EA_HID        264
#define EA_W1_STRIDE  96     // already a multiple of 16
#define EA_W2_STRIDE  272    // 264 columns padded up to a 16-byte stride
#define EA_LAYERS     8

#define EA_W1_BYTES   (EA_HID * EA_W1_STRIDE)
#define EA_W3_BYTES   (EA_HID * EA_W1_STRIDE)
#define EA_W2_BYTES   (EA_DIM * EA_W2_STRIDE)
#define EA_S1_BYTES   (EA_HID * (int)sizeof(float))
#define EA_S3_BYTES   (EA_HID * (int)sizeof(float))
#define EA_S2_BYTES   (EA_DIM * (int)sizeof(float))

#define EA_OFF_W1  0
#define EA_OFF_W3  (EA_OFF_W1 + EA_W1_BYTES)
#define EA_OFF_W2  (EA_OFF_W3 + EA_W3_BYTES)
#define EA_OFF_S1  (EA_OFF_W2 + EA_W2_BYTES)
#define EA_OFF_S3  (EA_OFF_S1 + EA_S1_BYTES)
#define EA_OFF_S2  (EA_OFF_S3 + EA_S3_BYTES)
#define EA_LAYER_BYTES  (EA_OFF_S2 + EA_S2_BYTES)
#define EA_EXPERT_BYTES (EA_LAYERS * EA_LAYER_BYTES)
#define EA_EXPERTS      2

#define EA_XS         (1.0f / 127.0f)
#define EA_CPU_MHZ    240
// Run 0 is cold (D-cache evicted); later runs are measured without an explicit
// eviction and are reported as "no-evict", not "warm": the 634 KB working set
// does not fit the 64 KB cache, so the expert is never fully cache-resident.
// Bump EA_RUNS for a stable comparison once a configuration looks promising.
#ifndef EA_RUNS
#define EA_RUNS       3
#endif

// 0: only run 0 has an eviction sweep.
//    EA_RUNS=10 gives one post-eviction sample and nine no-evict repeats.
// 1: evict before every run.
//    EA_RUNS=10 gives ten post-eviction samples.
#ifndef EA_COLD_EVERY
#define EA_COLD_EVERY 0
#endif
#define EA_EVICT_BYTES (128 * 1024)
#define EA_TILE_MAX   (16 * 1024)

static const int EA_TILES[] = {1024, 2048, 4096, 8192, 16384};
#define EA_NTILES ((int)(sizeof(EA_TILES) / sizeof(EA_TILES[0])))

typedef struct {
    const int8_t *w;
    const float *sc;
    float *out;
    int rows;
    int stride;
} ea_mat_t;

typedef struct {
    uint32_t copy_cyc;
    uint32_t comp_cyc;
    double acc;
} ea_result_t;

static inline uint32_t ea_cyc(void)
{
    return (uint32_t)esp_cpu_get_cycle_count();
}

static inline uint32_t ea_rand(uint32_t *s)
{
    uint32_t x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *s = x;
    return x;
}

// Read one byte per 32-byte line. Used to evict the D-cache between cold runs
// and, in the prefetch variant, to pull the next tile's lines in early.
static void ea_touch(const int8_t *p, size_t bytes)
{
    volatile int8_t sink = 0;
    for (size_t i = 0; i < bytes; i += 32) {
        sink ^= p[i];
    }
    (void)sink;
}

static void ea_put_f32(uint8_t *base, size_t off, float v)
{
    memcpy(base + off, &v, sizeof v);
}

static void ea_layer_mm(const int8_t *base, const int8_t *act,
                        float *o1, float *o2, float *o3)
{
    llm_mm_s8_generic(act, base + EA_OFF_W1,
                      (const float *)(base + EA_OFF_S1), EA_XS, o1,
                      EA_HID, EA_W1_STRIDE);
    llm_mm_s8_generic(act, base + EA_OFF_W3,
                      (const float *)(base + EA_OFF_S3), EA_XS, o2,
                      EA_HID, EA_W1_STRIDE);
    llm_mm_s8_generic(act, base + EA_OFF_W2,
                      (const float *)(base + EA_OFF_S2), EA_XS, o3,
                      EA_DIM, EA_W2_STRIDE);
}

static double ea_sum(const float *o1, const float *o2, const float *o3)
{
    double s = 0.0;
    for (int i = 0; i < EA_HID; i++) {
        s += (double)o1[i];
        s += (double)o2[i];
    }
    for (int i = 0; i < EA_DIM; i++) {
        s += (double)o3[i];
    }
    return s;
}

static const int8_t *ea_layer_base(const int8_t *psram, int pattern, int layer)
{
    size_t expert = pattern ? (size_t)(layer & 1) : 0;
    return psram + expert * EA_EXPERT_BYTES + (size_t)layer * EA_LAYER_BYTES;
}

// Mode A: whole per-layer slice copied into SRAM, then computed from SRAM.
static void ea_mode_a(const int8_t *psram, int8_t *sram, const int8_t *act,
                      float *o1, float *o2, float *o3, int pattern,
                      ea_result_t *r)
{
    memset(r, 0, sizeof *r);
    for (int l = 0; l < EA_LAYERS; l++) {
        const int8_t *src = ea_layer_base(psram, pattern, l);
        uint32_t c0 = ea_cyc();
        memcpy(sram, src, EA_LAYER_BYTES);
        uint32_t c1 = ea_cyc();
        ea_layer_mm(sram, act, o1, o2, o3);
        uint32_t c2 = ea_cyc();
        r->copy_cyc += c1 - c0;
        r->comp_cyc += c2 - c1;
        r->acc += ea_sum(o1, o2, o3);
    }
}

// Mode B: no copy, the kernels read the PSRAM mapping directly.
static void ea_mode_b(const int8_t *psram, const int8_t *act,
                      float *o1, float *o2, float *o3, int pattern,
                      ea_result_t *r)
{
    memset(r, 0, sizeof *r);
    for (int l = 0; l < EA_LAYERS; l++) {
        const int8_t *src = ea_layer_base(psram, pattern, l);
        uint32_t c0 = ea_cyc();
        ea_layer_mm(src, act, o1, o2, o3);
        uint32_t c1 = ea_cyc();
        r->comp_cyc += c1 - c0;
        r->acc += ea_sum(o1, o2, o3);
    }
}

static void ea_stream_mat(const ea_mat_t *mt, const int8_t *act,
                          int8_t *t0, int8_t *t1, int tile_bytes, bool prefetch,
                          ea_result_t *r)
{
    int rows_per_tile = tile_bytes / mt->stride;
    if (rows_per_tile < 1) {
        rows_per_tile = 1;
    }
    int flip = 0;
    for (int r0 = 0; r0 < mt->rows; r0 += rows_per_tile) {
        int n = mt->rows - r0;
        if (n > rows_per_tile) {
            n = rows_per_tile;
        }
        int8_t *buf = flip ? t1 : t0;
        flip ^= 1;
        const int8_t *src = mt->w + (size_t)r0 * (size_t)mt->stride;
        uint32_t c0 = ea_cyc();
        if (prefetch) {
            int nr = r0 + rows_per_tile;
            if (nr < mt->rows) {
                int nn = mt->rows - nr;
                if (nn > rows_per_tile) {
                    nn = rows_per_tile;
                }
                ea_touch(mt->w + (size_t)nr * (size_t)mt->stride,
                         (size_t)nn * (size_t)mt->stride);
            }
        }
        memcpy(buf, src, (size_t)n * (size_t)mt->stride);
        uint32_t c1 = ea_cyc();
        llm_mm_s8_generic(act, buf, mt->sc + r0, EA_XS, mt->out + r0, n,
                          mt->stride);
        uint32_t c2 = ea_cyc();
        r->copy_cyc += c1 - c0;
        r->comp_cyc += c2 - c1;
    }
}

// Element-wise agreement for one output matrix.
static void ea_cmp(const float *x, const float *y, int n, float *maxd, int *diff)
{
    float m = 0.0f;
    int d = 0;
    for (int i = 0; i < n; i++) {
        if (x[i] != y[i]) {
            d++;
        }
        float ad = fabsf(x[i] - y[i]);
        if (ad > m) {
            m = ad;
        }
    }
    *maxd = m;
    *diff = d;
}

// Element-wise equivalence on layer 0, per matrix. The aggregate accumulator
// in EA_RUN is only a sanity check: two different vectors can share a sum, so
// this compares every element of W1, W3 and W2 output separately.
static float eq_a1[EA_HID], eq_a2[EA_HID], eq_a3[EA_DIM];
static float eq_b1[EA_HID], eq_b2[EA_HID], eq_b3[EA_DIM];
static float eq_c1[EA_HID], eq_c2[EA_HID], eq_c3[EA_DIM];

static void ea_equiv_pass(const int8_t *psram, int8_t *sram,
                          int8_t *t0, int8_t *t1, const int8_t *act,
                          int tile_bytes)
{
    memcpy(sram, psram, EA_LAYER_BYTES);
    ea_layer_mm(sram, act, eq_a1, eq_a2, eq_a3);

    ea_layer_mm(psram, act, eq_b1, eq_b2, eq_b3);

    const int8_t *base = psram;
    ea_result_t sink;
    memset(&sink, 0, sizeof sink);
    ea_mat_t m1 = {base + EA_OFF_W1, (const float *)(base + EA_OFF_S1), eq_c1,
                   EA_HID, EA_W1_STRIDE};
    ea_mat_t m3 = {base + EA_OFF_W3, (const float *)(base + EA_OFF_S3), eq_c2,
                   EA_HID, EA_W1_STRIDE};
    ea_mat_t m2 = {base + EA_OFF_W2, (const float *)(base + EA_OFF_S2), eq_c3,
                   EA_DIM, EA_W2_STRIDE};
    ea_stream_mat(&m1, act, t0, t1, tile_bytes, false, &sink);
    ea_stream_mat(&m3, act, t0, t1, tile_bytes, false, &sink);
    ea_stream_mat(&m2, act, t0, t1, tile_bytes, false, &sink);

    float b1m, b2m, b3m, c1m, c2m, c3m;
    int b1d, b2d, b3d, c1d, c2d, c3d;
    ea_cmp(eq_a1, eq_b1, EA_HID, &b1m, &b1d);
    ea_cmp(eq_a2, eq_b2, EA_HID, &b2m, &b2d);
    ea_cmp(eq_a3, eq_b3, EA_DIM, &b3m, &b3d);
    ea_cmp(eq_a1, eq_c1, EA_HID, &c1m, &c1d);
    ea_cmp(eq_a2, eq_c2, EA_HID, &c2m, &c2d);
    ea_cmp(eq_a3, eq_c3, EA_DIM, &c3m, &c3d);

    printf("EA_EQUIV pair=a_b ok=%d w1_max=%.6f w1_diff=%d w3_max=%.6f "
           "w3_diff=%d w2_max=%.6f w2_diff=%d\n",
           b1d == 0 && b2d == 0 && b3d == 0, b1m, b1d, b2m, b2d, b3m, b3d);
    printf("EA_EQUIV pair=a_c ok=%d w1_max=%.6f w1_diff=%d w3_max=%.6f "
           "w3_diff=%d w2_max=%.6f w2_diff=%d\n",
           c1d == 0 && c2d == 0 && c3d == 0, c1m, c1d, c2m, c2d, c3m, c3d);
}

// Mode C: bounded tiles, alternating between two SRAM buffers.
static void ea_mode_c(const int8_t *psram, int8_t *t0, int8_t *t1,
                      const int8_t *act, float *o1, float *o2, float *o3,
                      int pattern, int tile_bytes, bool prefetch,
                      ea_result_t *r)
{
    memset(r, 0, sizeof *r);
    for (int l = 0; l < EA_LAYERS; l++) {
        const int8_t *base = ea_layer_base(psram, pattern, l);
        ea_mat_t mats[3] = {
            {base + EA_OFF_W1, (const float *)(base + EA_OFF_S1), o1,
             EA_HID, EA_W1_STRIDE},
            {base + EA_OFF_W3, (const float *)(base + EA_OFF_S3), o2,
             EA_HID, EA_W1_STRIDE},
            {base + EA_OFF_W2, (const float *)(base + EA_OFF_S2), o3,
             EA_DIM, EA_W2_STRIDE},
        };
        for (int mi = 0; mi < 3; mi++) {
            ea_stream_mat(&mats[mi], act, t0, t1, tile_bytes, prefetch, r);
        }
        r->acc += ea_sum(o1, o2, o3);
    }
}

static void ea_print(const char *mode, const char *pattern, int run, int tile,
                     const ea_result_t *r, uint64_t psram_bytes)
{
    uint32_t total = r->copy_cyc + r->comp_cyc;
    double us = (double)total / (double)EA_CPU_MHZ;
    double mbps = us > 0.0 ? (double)psram_bytes / us : 0.0;
    // Emit the state explicitly: with EA_COLD_EVERY=1 every run is cold, so
    // "run == 0" alone is not a reliable cold marker for the analyser.
    int cold = (run == 0 || EA_COLD_EVERY) ? 1 : 0;
    printf("EA_RUN mode=%s pattern=%s run=%d cold=%d tile=%d copy_cyc=%u "
           "comp_cyc=%u total_cyc=%u copy_us=%u comp_us=%u total_us=%u "
           "psram_bytes=%llu mbps=%.2f acc=%.6f\n",
           mode, pattern, run, cold, tile,
           (unsigned)r->copy_cyc, (unsigned)r->comp_cyc, (unsigned)total,
           (unsigned)(r->copy_cyc / EA_CPU_MHZ),
           (unsigned)(r->comp_cyc / EA_CPU_MHZ),
           (unsigned)us,
           (unsigned long long)psram_bytes, mbps, r->acc);
}

// Stride-64 control: the only geometry that selects llm_mm_s8_64(). Reports the
// specialised kernel against the generic one, from SRAM and from PSRAM.
static void ea_control64(int8_t *sram, uint32_t *seed)
{
    enum { C64_ROWS = 256, C64_STRIDE = 64 };
    const size_t bytes = (size_t)C64_ROWS * C64_STRIDE;
    int8_t *wp = heap_caps_aligned_alloc(16, bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    int8_t *ws = sram;
    static float sc64[C64_ROWS];
    static float out_fast[C64_ROWS];
    static float out_gen[C64_ROWS];
    if (!wp || bytes > EA_LAYER_BYTES) {
        printf("EA_CTRL64 rc=alloc\n");
        free(wp);
        return;
    }
    int8_t act64[64] __attribute__((aligned(16)));
    for (size_t i = 0; i < bytes; i++) {
        wp[i] = (int8_t)((int)(ea_rand(seed) & 0xFF) - 128);
    }
    for (int i = 0; i < C64_ROWS; i++) {
        sc64[i] = 0.001f + 0.0005f * (float)(i & 7);
    }
    for (int i = 0; i < 64; i++) {
        act64[i] = (int8_t)((int)(ea_rand(seed) & 0xFF) - 128);
    }
    memcpy(ws, wp, bytes);

    float xs = EA_XS;
    uint32_t t0 = ea_cyc();
    llm_mm_s8_64(act64, ws, sc64, &xs, out_fast, C64_ROWS);
    uint32_t t1 = ea_cyc();
    llm_mm_s8_generic(act64, ws, sc64, xs, out_gen, C64_ROWS, C64_STRIDE);
    uint32_t t2 = ea_cyc();
    llm_mm_s8_64(act64, wp, sc64, &xs, out_fast, C64_ROWS);
    uint32_t t3 = ea_cyc();
    llm_mm_s8_generic(act64, wp, sc64, xs, out_gen, C64_ROWS, C64_STRIDE);
    uint32_t t4 = ea_cyc();

    float maxd = 0.0f;
    for (int i = 0; i < C64_ROWS; i++) {
        float d = fabsf(out_fast[i] - out_gen[i]);
        if (d > maxd) {
            maxd = d;
        }
    }
    printf("EA_CTRL64 rows=%d stride=%d sram_fast_cyc=%u sram_generic_cyc=%u "
           "psram_fast_cyc=%u psram_generic_cyc=%u max_diff=%.6f\n",
           C64_ROWS, C64_STRIDE, (unsigned)(t1 - t0), (unsigned)(t2 - t1),
           (unsigned)(t3 - t2), (unsigned)(t4 - t3), maxd);
    free(wp);
}

void expert_access_bench_run(void)
{
    printf("EA_GEOM dim=%d hidden=%d layers=%d w1_stride=%d w2_stride=%d "
           "layer_bytes=%d expert_bytes=%d experts=%d\n",
           EA_DIM, EA_HID, EA_LAYERS, EA_W1_STRIDE, EA_W2_STRIDE,
           EA_LAYER_BYTES, EA_EXPERT_BYTES, EA_EXPERTS);
    printf("EA_MEM psram_free=%u psram_largest=%u internal_free=%u "
           "internal_largest=%u\n",
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
           (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM),
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
           (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL));

    const size_t expert_total = (size_t)EA_EXPERTS * EA_EXPERT_BYTES;
    uint8_t *psram = heap_caps_aligned_alloc(16, expert_total,
                                             MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    uint8_t *evict = heap_caps_malloc(EA_EVICT_BYTES,
                                      MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    int8_t *sram = heap_caps_aligned_alloc(16, EA_LAYER_BYTES,
                                           MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    int8_t *tile = heap_caps_aligned_alloc(16, 2 * EA_TILE_MAX,
                                           MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    int8_t *act = heap_caps_aligned_alloc(16, EA_W2_STRIDE,
                                          MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    float *o1 = heap_caps_malloc(EA_HID * sizeof(float), MALLOC_CAP_INTERNAL);
    float *o2 = heap_caps_malloc(EA_HID * sizeof(float), MALLOC_CAP_INTERNAL);
    float *o3 = heap_caps_malloc(EA_DIM * sizeof(float), MALLOC_CAP_INTERNAL);

    if (!psram || !evict || !sram || !tile || !act || !o1 || !o2 || !o3) {
        printf("EA_ERR alloc psram=%p evict=%p sram=%p tile=%p act=%p "
               "o1=%p o2=%p o3=%p\n",
               (void *)psram, (void *)evict, (void *)sram, (void *)tile,
               (void *)act, (void *)o1, (void *)o2, (void *)o3);
        printf("EA_END\n");
        fflush(stdout);
        free(psram);
        free(evict);
        free(sram);
        free(tile);
        free(act);
        free(o1);
        free(o2);
        free(o3);
        return;
    }

    // heap_caps_malloc() is not zeroed. Initialise the eviction buffer so the
    // sweep reads defined data; its purpose is only to displace cache lines.
    memset(evict, 0xA5, EA_EVICT_BYTES);

    uint32_t seed = 0x12345678u;
    for (int e = 0; e < EA_EXPERTS; e++) {
        for (int l = 0; l < EA_LAYERS; l++) {
            uint8_t *base = psram + (size_t)e * EA_EXPERT_BYTES +
                            (size_t)l * EA_LAYER_BYTES;
            for (int i = 0; i < EA_OFF_S1; i++) {
                base[i] = (uint8_t)(int8_t)((int)(ea_rand(&seed) & 0xFF) - 128);
            }
            for (int i = 0; i < EA_HID; i++) {
                ea_put_f32(base, (size_t)EA_OFF_S1 + (size_t)i * 4,
                           0.001f + 0.0005f * (float)(i & 7));
                ea_put_f32(base, (size_t)EA_OFF_S3 + (size_t)i * 4,
                           0.001f + 0.0005f * (float)((i + 3) & 7));
            }
            for (int i = 0; i < EA_DIM; i++) {
                ea_put_f32(base, (size_t)EA_OFF_S2 + (size_t)i * 4,
                           0.001f + 0.0005f * (float)((i + 5) & 7));
            }
        }
    }
    for (int i = 0; i < EA_DIM; i++) {
        act[i] = (int8_t)((int)(ea_rand(&seed) & 0xFF) - 128);
    }
    memset(act + EA_DIM, 0, (size_t)(EA_W2_STRIDE - EA_DIM));

    const char *patterns[2] = {"pinned", "alternating"};
    double acc_a = 0.0, acc_b = 0.0, acc_c = 0.0;

    for (int p = 0; p < 2; p++) {
        for (int run = 0; run < EA_RUNS; run++) {
            ea_result_t r;
            if (run == 0 || EA_COLD_EVERY) {
                ea_touch((const int8_t *)evict, EA_EVICT_BYTES);
            }
            ea_mode_a((const int8_t *)psram, sram, act, o1, o2, o3, p, &r);
            ea_print("A", patterns[p], run, 0, &r, EA_EXPERT_BYTES);
            if (p == 0 && run == 0) {
                acc_a = r.acc;
            }
        }
        for (int run = 0; run < EA_RUNS; run++) {
            ea_result_t r;
            if (run == 0 || EA_COLD_EVERY) {
                ea_touch((const int8_t *)evict, EA_EVICT_BYTES);
            }
            ea_mode_b((const int8_t *)psram, act, o1, o2, o3, p, &r);
            ea_print("B", patterns[p], run, 0, &r, EA_EXPERT_BYTES);
            if (p == 0 && run == 0) {
                acc_b = r.acc;
            }
        }
        for (int ti = 0; ti < EA_NTILES; ti++) {
            for (int pf = 0; pf < 2; pf++) {
                for (int run = 0; run < EA_RUNS; run++) {
                    ea_result_t r;
                    if (run == 0 || EA_COLD_EVERY) {
                        ea_touch((const int8_t *)evict, EA_EVICT_BYTES);
                    }
                    ea_mode_c((const int8_t *)psram, tile, tile + EA_TILE_MAX,
                              act, o1, o2, o3, p, EA_TILES[ti], pf != 0, &r);
                    ea_print(pf ? "C-prefetch" : "C", patterns[p], run,
                             EA_TILES[ti], &r, EA_EXPERT_BYTES);
                    if (p == 0 && run == 0 && ti == 2 && pf == 0) {
                        acc_c = r.acc;
                    }
                }
            }
        }
    }

    printf("EA_ACC a=%.6f b=%.6f c=%.6f equal_ab=%d equal_ac=%d\n",
           acc_a, acc_b, acc_c, acc_a == acc_b, acc_a == acc_c);
    ea_equiv_pass((const int8_t *)psram, sram, tile, tile + EA_TILE_MAX, act,
                  4096);

    ea_control64(sram, &seed);

    printf("EA_END\n");
    fflush(stdout);

    free(psram);
    free(evict);
    free(sram);
    free(tile);
    free(act);
    free(o1);
    free(o2);
    free(o3);
}
