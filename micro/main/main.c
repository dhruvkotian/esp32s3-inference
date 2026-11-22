// Vendor versus same-tree int8 dot kernels, timed on the QK^T shape.
//
// Everything here runs from internal SRAM with a 16 KB working set, so the
// numbers are instruction cost, not PSRAM traffic. The paper's scan numbers
// come from the main firmware; this app exists only to answer one question:
// what does ESP-DSP's 16-lane int8 dot cost next to ours, per block and per
// four-position batch.
//
// Buffers are laid out like the real cache: a K record is 32 bytes and holds
// four KV heads of 8 values, and the query is one head padded with zeros, so
// a 16-lane dot covers one head and cancels the neighbour.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "esp_cpu.h"
#include "esp_dsp.h"
#include "sdkconfig.h"

int32_t llm_dot_s8(const int8_t *a, const int8_t *b, int n);
int32_t llm_dot_s8_scalar(const int8_t *a, const int8_t *b, int n);
int32_t llm_dot16_s8(const int8_t *a, const int8_t *b);
void llm_dot16x4_s8(const int8_t *a, const int8_t *b, int stride, int32_t out[4]);

#define ESP_DSP_TAG "v1.8.2"
#define POSITIONS 496          // the scan length at 512 context
#define KSTRIDE   32           // one K record: four KV heads of 8 int8
#define KVBYTES   (POSITIONS * KSTRIDE)
#define TRIALS    5

static int8_t q[16] __attribute__((aligned(16)));
static int8_t kbig[KVBYTES] __attribute__((aligned(16)));
static int8_t vflat[POSITIONS] __attribute__((aligned(16)));
static int8_t vflat2[POSITIONS] __attribute__((aligned(16)));
static int32_t out4[4];

static volatile int32_t sink;

// A call that does nothing, to price the loop and call overhead.
static __attribute__((noinline)) int32_t empty_call(void)
{
    return 0;
}

static uint32_t now(void)
{
    return esp_cpu_get_cycle_count();
}

static int cmp_u32(const void *a, const void *b)
{
    uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
    return (x > y) - (x < y);
}

// Report the minimum and the median in cycles per call, scaled by 1000 so
// sub-cycle differences are visible.
static void report(const char *name, const uint32_t *cycles, int reps, int blocks)
{
    uint32_t sorted[TRIALS];
    memcpy(sorted, cycles, sizeof(sorted));
    qsort(sorted, TRIALS, sizeof(uint32_t), cmp_u32);
    uint32_t mn = sorted[0], med = sorted[TRIALS / 2];
    // Raw cycle counts; the host divides by reps and blocks. newlib nano does
    // not print 64-bit integers, and the totals fit in 32 bits anyway.
    printf("CASE %-24s reps=%-6d blocks=%-3d cyc_min=%lu cyc_med=%lu\n",
           name, reps, blocks, (unsigned long)mn, (unsigned long)med);
}

static void run(const char *name, void (*fn)(void), int blocks, int reps)
{
    uint32_t cyc[TRIALS];
    for (int t = 0; t < TRIALS; t++) {
        uint32_t t0 = now();
        for (int i = 0; i < reps; i++) {
            fn();
        }
        cyc[t] = now() - t0;
    }
    report(name, cyc, reps, blocks);
}

static int32_t scalar_ref(const int8_t *a, const int8_t *b, int n)
{
    int32_t s = 0;
    for (int i = 0; i < n; i++) {
        s += (int32_t)a[i] * (int32_t)b[i];
    }
    return s;
}

static int check(const char *name, int32_t got, int32_t want)
{
    int ok = got == want;
    printf("CHECK %-28s got=%-8d want=%-8d %s\n", name, (int)got, (int)want,
           ok ? "PASS" : "FAIL");
    return ok;
}

// ---- one operation per function, so the timed loop has no macro tricks ----

static void c_empty(void) { sink = empty_call(); }

static void c_llm_dot16(void)
{
    sink = llm_dot16_s8(q, kbig);
}

static void c_dsp_len16(void)
{
    int32_t o;
    dsps_dp_s8(q, kbig, &o, 16);
    sink = o;
}

static void c_llm_len32(void) { sink = llm_dot_s8(q, kbig, 32); }

static void c_dsp_len32(void)
{
    int32_t o;
    dsps_dp_s8(q, kbig, &o, 32);
    sink = o;
}

static void c_llm_len128(void) { sink = llm_dot_s8(q, kbig, 128); }

static void c_dsp_len128(void)
{
    int32_t o;
    dsps_dp_s8(q, kbig, &o, 128);
    sink = o;
}

static void c_llm_len496(void) { sink = llm_dot_s8(vflat, vflat2, POSITIONS); }

static void c_dsp_len496(void)
{
    int32_t o;
    dsps_dp_s8(vflat, vflat2, &o, POSITIONS);
    sink = o;
}

static void c_llm_x4(void)
{
    llm_dot16x4_s8(q, kbig, KSTRIDE, out4);
    sink = out4[0] + out4[1] + out4[2] + out4[3];
}

static void c_llm_x1_x4(void)
{
    int32_t s = llm_dot16_s8(q, kbig);
    s += llm_dot16_s8(q, kbig + KSTRIDE);
    s += llm_dot16_s8(q, kbig + 2 * KSTRIDE);
    s += llm_dot16_s8(q, kbig + 3 * KSTRIDE);
    sink = s;
}

static void c_dsp_x1_x4(void)
{
    int32_t o0, o1, o2, o3;
    dsps_dp_s8(q, kbig, &o0, 16);
    dsps_dp_s8(q, kbig + KSTRIDE, &o1, 16);
    dsps_dp_s8(q, kbig + 2 * KSTRIDE, &o2, 16);
    dsps_dp_s8(q, kbig + 3 * KSTRIDE, &o3, 16);
    sink = o0 + o1 + o2 + o3;
}

void app_main(void)
{
    // Deterministic, mixed-sign data.
    uint32_t seed = 0x12345678u;
    for (int i = 0; i < KVBYTES; i++) {
        seed = seed * 1664525u + 1013904223u;
        kbig[i] = (int8_t)((seed >> 16) & 0xff);
    }
    for (int i = 0; i < 16; i++) {
        seed = seed * 1664525u + 1013904223u;
        q[i] = (int8_t)((seed >> 16) & 0xff);
    }
    // Zero the second half of the query, as the attention kernel does, so the
    // dot covers one head of the 32-byte K record.
    memset(q + 8, 0, 8);
    for (int i = 0; i < POSITIONS; i++) {
        seed = seed * 1664525u + 1013904223u;
        vflat[i] = (int8_t)((seed >> 16) & 0xff);
        seed = seed * 1664525u + 1013904223u;
        vflat2[i] = (int8_t)((seed >> 16) & 0xff);
    }

    printf("\nMICRO start\n");
    printf("MICRO esp-dsp %s\n", ESP_DSP_TAG);
    printf("MICRO cpu %d MHz, positions %d, kstride %d, working set %d B\n",
           CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ, POSITIONS, KSTRIDE, KVBYTES);

    // ---- correctness -----------------------------------------------------
    int ok = 1;
    for (int p = 0; p < POSITIONS; p += 97) {
        int32_t want = scalar_ref(q, kbig + (size_t)p * KSTRIDE, 16);
        ok &= check("llm_dot16_s8", llm_dot16_s8(q, kbig + (size_t)p * KSTRIDE), want);
        int32_t v = 0;
        esp_err_t e = dsps_dp_s8(q, kbig + (size_t)p * KSTRIDE, &v, 16);
        if (e != ESP_OK) {
            printf("CHECK dsps_dp_s8 len=16 error=%d FAIL\n", (int)e);
            ok = 0;
        }
        ok &= check("dsps_dp_s8 len=16", v, want);
    }

    int32_t want32 = scalar_ref(q, kbig, 32);
    ok &= check("llm_dot_s8 len=32", llm_dot_s8(q, kbig, 32), want32);
    int32_t v32 = 0;
    dsps_dp_s8(q, kbig, &v32, 32);
    ok &= check("dsps_dp_s8 len=32", v32, want32);

    int32_t want128 = scalar_ref(q, kbig, 128);
    ok &= check("llm_dot_s8 len=128", llm_dot_s8(q, kbig, 128), want128);
    int32_t v128 = 0;
    dsps_dp_s8(q, kbig, &v128, 128);
    ok &= check("dsps_dp_s8 len=128", v128, want128);

    int32_t want496 = scalar_ref(vflat, vflat2, POSITIONS);
    ok &= check("llm_dot_s8 len=496", llm_dot_s8(vflat, vflat2, POSITIONS), want496);
    int32_t v496 = 0;
    dsps_dp_s8(vflat, vflat2, &v496, POSITIONS);
    ok &= check("dsps_dp_s8 len=496", v496, want496);

    llm_dot16x4_s8(q, kbig, KSTRIDE, out4);
    for (int j = 0; j < 4; j++) {
        char label[32];
        snprintf(label, sizeof(label), "llm_dot16x4_s8[%d]", j);
        ok &= check(label, out4[j], scalar_ref(q, kbig + (size_t)j * KSTRIDE, 16));
    }
    printf("MICRO correctness %s\n", ok ? "PASS" : "FAIL");

    // ---- timing ----------------------------------------------------------
    run("empty_call", c_empty, 0, 200000);
    run("llm_dot16_s8", c_llm_dot16, 1, 200000);
    run("dsps_dp_s8_len16", c_dsp_len16, 1, 200000);
    run("llm_dot_s8_len32", c_llm_len32, 2, 100000);
    run("dsps_dp_s8_len32", c_dsp_len32, 2, 100000);
    run("llm_dot_s8_len128", c_llm_len128, 8, 50000);
    run("dsps_dp_s8_len128", c_dsp_len128, 8, 50000);
    run("llm_dot_s8_len496", c_llm_len496, 31, 20000);
    run("dsps_dp_s8_len496", c_dsp_len496, 31, 20000);
    run("llm_dot16x4_s8", c_llm_x4, 4, 50000);
    run("four_llm_dot16_s8", c_llm_x1_x4, 4, 50000);
    run("four_dsps_dp_s8_len16", c_dsp_x1_x4, 4, 50000);

    printf("MICRO done\n");
}
