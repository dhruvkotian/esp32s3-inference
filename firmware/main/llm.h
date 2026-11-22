#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Per-row int8 quantized matrix. Rows are padded to a 16-byte stride so the
// 128-bit loads in dot_s8.S are always aligned.
typedef struct {
    int8_t *q;
    float *scale;
    int rows;
    int cols;
    int stride;
} qmat_t;

typedef struct {
    // model config (legacy llama2.c header)
    int dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len;
    int kv_dim, head_size, kv_mul;

    // whole model file, held in PSRAM
    uint8_t *raw;
    size_t raw_size;

    // fp32 views into raw (norms are kept in fp32; matrices are also kept
    // here so the fp32 reference path can run on the same buffer)
    const float *f_tok_emb, *f_rms_att, *f_wq, *f_wk, *f_wv, *f_wo;
    const float *f_rms_ffn, *f_w1, *f_w2, *f_w3, *f_rms_final;

    // int8 copies of the 2D weight matrices
    qmat_t tok_emb, wq, wk, wv, wo, w1, w2, w3;
    bool int8_ready;

    // run state
    int ctx;
    float *x, *xb, *xb2, *qbuf, *hb, *hb2, *att;
    float *key_cache, *value_cache;
    // INT8 KV experiment (KV_INT8 in bench_cfg.h). K/V live in PSRAM as one
    // byte per element with one fp32 scale per head per position. Scales are
    // [layer][kv_head][pos], so a head reads them sequentially. The hot window
    // below stays fp32, so recent positions keep full precision.
    int8_t *key_q, *value_q;
    float *key_scale, *value_scale;
    float *ktmp, *vtmp;      // fp32 staging for the current position
    float *kdeq, *vdeq;      // dequantised head, one position at a time
    int8_t *qpad;            // per-head query, 16-byte padded, int8
    float *qscale;           // per-head query scale
    int8_t *a8;              // folded softmax weights for the V dot
    // Dual-core attention split (DUAL_CORE in bench_cfg.h). The worker core
    // runs the upper half of the heads with its own scratch buffers.
    float *att2;
    int8_t *a8_2;
    void *attn_task;         // worker task handle (opaque here)
    void *attn_main;         // calling task handle, for the done notification
    int attn_l;
    int attn_pos;
    bool attn_q8;
    bool dc_ready;
    // DMA V prefetch (DMA_PREFETCH in bench_cfg.h).
    void *dma;               // async memcpy handle
    int8_t *vbuf0;
    int8_t *vbuf1;
    volatile int vdone0;
    volatile int vdone1;
    // Most recent `hot` positions of K/V mirrored in internal SRAM. A PSRAM
    // read costs 8x an SRAM read here (measured 1174 vs 139 ns per 8-float
    // dot), and the loads block, so no prefetching helps. Older positions
    // still come from PSRAM, which keeps long contexts correct.
    float *khot, *vhot;
    int hot;
    float *logits;
    int8_t *qx;
    int stride_max;
    // Precomputed rotary table: cos/sin for every (position, dim/2) pair.
    // powf/cosf/sinf are libm calls on this chip (the FPU only does add, mul,
    // FMA and convert), so recomputing them inside the forward pass costs more
    // than the vector MACs.
    float *rope_cos, *rope_sin;
    int rope_stride;

    // Per-section timing, accumulated over int8 forward passes only.
    uint64_t prof[8];
    // Attention sub-profile in CPU cycles, one set per core.
    // [0] K score, [1] softmax, [2] V weight quant, [3] V accumulate.
    uint64_t aprof[4];
    uint64_t aprof2[4];
} llm_t;

// Parse a llama2.c v0 model already resident in memory.
int llm_parse(llm_t *m, uint8_t *raw, size_t size, int ctx);

// Build the int8 copies from the fp32 tensors.
int llm_quantize(llm_t *m, bool prefer_internal);

int llm_forward_f32(llm_t *m, int token, int pos, float *logits);
int llm_forward_int8(llm_t *m, int token, int pos, float *logits);
// Free the fp32 K/V caches after the last fp32 reference pass. Only safe when
// no further fp32 forward will run; the int8 path never reads them.
void llm_drop_fp32_cache(llm_t *m);

size_t llm_int8_bytes(const llm_t *m);
void llm_print_mem(const llm_t *m);

// SIMD kernels (dot_s8.S)
int32_t llm_dot_s8(const int8_t *a, const int8_t *b, int n);
int32_t llm_dot_s8_scalar(const int8_t *a, const int8_t *b, int n);
int32_t llm_dot_s8_fused64(const int8_t *a, const int8_t *b);
int32_t llm_dot16_s8(const int8_t *a, const int8_t *b);
void llm_dot16x4_s8(const int8_t *a, const int8_t *b, int stride, int32_t out[4]);
void llm_mm_s8_64(const int8_t *act, const int8_t *w, const float *scale,
                  const float *xs, float *out, int nrows);