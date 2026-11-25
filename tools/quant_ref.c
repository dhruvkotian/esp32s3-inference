/* Host reference for the firmware's int8 weight quantizer.
 *
 * The function below is a verbatim copy of qmat_fill() from
 * firmware/main/llm.c (lines 492-517) plus round_to_int() (lines 69-72).
 * It is compiled on the host only so tools/pack_model_m8.py can prove that
 * its numpy quantizer produces byte-identical q bytes and scales.
 *
 * Build:
 *   gcc -O2 -ffp-contract=off -o build/quant_ref tools/quant_ref.c
 * Run:
 *   build/quant_ref model/stories260K.bin build/quant_ref.bin
 *
 * Output layout ("Q8REF1\0\0", u32 count, then per tensor
 * u32 rows, u32 cols, u32 stride, q[rows*stride], f32 scale[rows]):
 *   1 tok_embeddings (vocab x dim)
 *   2 wq             (nl*dim x dim)
 *   3 wk             (nl*kv  x dim)
 *   4 wv             (nl*kv  x dim)
 *   5 wo             (nl*dim x dim)
 *   6 w1             (nl*hid x dim)
 *   7 w2             (nl*dim x hid)
 *   8 w3             (nl*hid x dim)
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- verbatim from firmware/main/llm.c ---- */
static inline int round_to_int(float x)
{
    return (int)(x + (x >= 0.0f ? 0.5f : -0.5f));
}

typedef struct {
    int8_t *q;
    float *scale;
    int rows;
    int cols;
    int stride;
} qmat_t;

static void qmat_fill(qmat_t *q, const float *src)
{
    for (int r = 0; r < q->rows; r++) {
        const float *row = src + (size_t)r * q->cols;
        float mx = 0.0f;
        for (int c = 0; c < q->cols; c++) {
            float a = row[c] < 0.0f ? -row[c] : row[c];
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
/* ---- end verbatim (fabsf replaced by a branch, same result) ---- */

static uint8_t *read_file(const char *path, size_t *size)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        perror(path);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n <= 0) {
        fclose(f);
        return NULL;
    }
    uint8_t *buf = malloc((size_t)n);
    if (!buf || fread(buf, 1, (size_t)n, f) != (size_t)n) {
        free(buf);
        fclose(f);
        return NULL;
    }
    fclose(f);
    *size = (size_t)n;
    return buf;
}

static void put_u32(FILE *f, uint32_t v)
{
    uint8_t b[4] = { (uint8_t)v, (uint8_t)(v >> 8), (uint8_t)(v >> 16), (uint8_t)(v >> 24) };
    fwrite(b, 1, 4, f);
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: %s model.bin out.bin\n", argv[0]);
        return 2;
    }
    size_t size = 0;
    uint8_t *raw = read_file(argv[1], &size);
    if (!raw) {
        return 1;
    }
    if (size < 28) {
        fprintf(stderr, "short header\n");
        return 1;
    }

    int32_t hdr[7];
    memcpy(hdr, raw, sizeof(hdr));
    const int dim = hdr[0], hid = hdr[1], nl = hdr[2];
    const int nh = hdr[3], nkvh = hdr[4], vocab = hdr[5], seq = hdr[6];
    if (vocab <= 0 || dim <= 0 || nh <= 0 || nkvh <= 0) {
        fprintf(stderr, "unsupported config\n");
        return 1;
    }
    const int kv = dim * nkvh / nh;
    const int hs = dim / nh;

    size_t off = 28, o;
    const float *f_tok, *f_wq, *f_wk, *f_wv, *f_wo, *f_w1, *f_w2, *f_w3;
#define TAKE(n, dst) do { \
        if (off + (size_t)(n) * 4 > size) { fprintf(stderr, "truncated\n"); return 1; } \
        (dst) = (const float *)(raw + off); off += (size_t)(n) * 4; \
    } while (0)
    TAKE((size_t)vocab * dim, f_tok);
    off += (size_t)nl * dim * 4;                 /* rms_att, not quantized */
    TAKE((size_t)nl * dim * dim, f_wq);
    TAKE((size_t)nl * kv * dim, f_wk);
    TAKE((size_t)nl * kv * dim, f_wv);
    TAKE((size_t)nl * dim * dim, f_wo);
    off += (size_t)nl * dim * 4;                 /* rms_ffn */
    TAKE((size_t)nl * hid * dim, f_w1);
    TAKE((size_t)nl * hid * dim, f_w2);
    TAKE((size_t)nl * hid * dim, f_w3);
    (void)o; (void)hs; (void)seq;
#undef TAKE

    struct spec { const float *src; int rows, cols; } specs[8] = {
        { f_tok, vocab, dim },
        { f_wq, nl * dim, dim },
        { f_wk, nl * kv, dim },
        { f_wv, nl * kv, dim },
        { f_wo, nl * dim, dim },
        { f_w1, nl * hid, dim },
        { f_w2, nl * dim, hid },
        { f_w3, nl * hid, dim },
    };

    FILE *out = fopen(argv[2], "wb");
    if (!out) {
        perror(argv[2]);
        return 1;
    }
    fwrite("Q8REF1\0\0", 1, 8, out);
    put_u32(out, 8);

    size_t total = 0;
    for (int i = 0; i < 8; i++) {
        qmat_t q;
        q.rows = specs[i].rows;
        q.cols = specs[i].cols;
        q.stride = (q.cols + 15) & ~15;
        q.q = calloc((size_t)q.rows * q.stride, 1);
        q.scale = calloc((size_t)q.rows, sizeof(float));
        if (!q.q || !q.scale) {
            fprintf(stderr, "oom\n");
            return 1;
        }
        qmat_fill(&q, specs[i].src);
        put_u32(out, (uint32_t)q.rows);
        put_u32(out, (uint32_t)q.cols);
        put_u32(out, (uint32_t)q.stride);
        fwrite(q.q, 1, (size_t)q.rows * q.stride, out);
        fwrite(q.scale, sizeof(float), (size_t)q.rows, out);
        total += (size_t)q.rows * q.stride + (size_t)q.rows * 4;
        free(q.q);
        free(q.scale);
    }
    fclose(out);
    fprintf(stderr, "wrote %s: 8 tensors, %zu B payload\n", argv[2], total);
    free(raw);
    return 0;
}
