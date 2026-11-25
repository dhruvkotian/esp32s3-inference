// ESP32-S3 on-device LLM experiment.
//
// Goals:
//   1. load a TinyStories llama2.c model from the SD card (camera never touched)
//   2. run it with int8 weights + 128-bit vector dot products
//   3. measure tokens/s at 240 MHz, then retry with a 320 MHz CPU overclock
//
// The only card write is staging /sdcard/models/default.m8 from the flash
// model partition when that file is absent. Flash is otherwise read-only.

#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "esp_chip_info.h"
#include "esp_private/esp_clk.h"
#include "esp_flash.h"
#include "esp_idf_version.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_memory_utils.h"
#include "esp_partition.h"
#include "esp_psram.h"
#include "esp_rom_sys.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "esp_task_wdt.h"
#include "esp_vfs_fat.h"
#include "driver/sdmmc_host.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "hal/clk_tree_ll.h"
#include "hal/clk_tree_hal.h"
#include "hal/regi2c_ctrl_ll.h"
#include "sdmmc_cmd.h"
#include "soc/rtc.h"
#include "soc/system_reg.h"

#include "llm.h"
#include "tokenizer.h"
#include "model_lib.h"
#include "model_prefs.h"
#include "bench_cfg.h"

static const char *TAG = "llms3";

// Active stage cache for runtime switching. It holds the active blob in PSRAM
// so a failed switch can roll back without touching the card.
static model_ctx_t g_model;

#define SD_MMC_CMD 38
#define SD_MMC_CLK 39
#define SD_MMC_D0  40
#define MODEL_PATH "/sdcard/stories260K.bin"
#define TOK_PATH   "/sdcard/tok512.bin"
#define MODEL_DIR  "/sdcard/models"
#define M8_PATH    "/sdcard/models/default.m8"
#define SWITCH_ALT_PATH "/sdcard/models/alt.m8"

// Model-library test harness (temporary; off in the published build).
//   0  off
//   1  provision alt.m8 from the partition, then select alt -> default -> alt
//   2  report the boot-selected model and exit (run after 1 to see NVS)
//   3  select default.m8, persist it and reboot; on the next boot confirm it
#ifndef LIB_TEST
#define LIB_TEST 0
#endif
#define MODEL_LIST_MAX 12

// Runtime-switch harness. Off in normal builds; the test build enables it in
// bench_cfg.h. M8_TEST_PARTITION_ONLY is test scaffolding too: it boots the
// partition blob directly, used to capture a new model's golden logits.
#ifndef SWITCH_TEST
#define SWITCH_TEST 0
#endif
#ifndef SWITCH_RUNS
#define SWITCH_RUNS 100
#endif
#ifndef M8_TEST_PARTITION_ONLY
#define M8_TEST_PARTITION_ONLY 0
#endif

// 0 means "use the model's own seq_len" (512 for stories260K). The KV cache is
// sized from this, so there is no reason to cap it below the trained context.
#define CTX_TOKENS 0
#define GEN_TOKENS 12
#define SOAK_RUNS  5

// fp32 greedy output for "Once upon a time", cross-checked against llama2.c
// run.c (5500 tok/s on the laptop) and tools/ref_llama.py.
static const int EXPECTED[] = {432, 383, 286, 261, 376, 298, 315, 421, 395, 317, 426, 338};
static const int EXPECTED_N = (int)(sizeof(EXPECTED) / sizeof(EXPECTED[0]));

typedef struct {
    int n_prompt;
    int n_gen;
    int ids[64];
    char text[768];
    int64_t ttft_us;
    int64_t total_us;
} gen_result_t;

static void banner(void)
{
    esp_chip_info_t info;
    esp_chip_info(&info);
    uint32_t flash = 0;
    esp_flash_get_size(NULL, &flash);
    ESP_LOGI(TAG, "==============================================");
    ESP_LOGI(TAG, " ParoClaw ESP32-S3 on-device LLM experiment");
    ESP_LOGI(TAG, " chip=%s cores=%d rev=%d.%d flash=%uMB psram=%uMB",
             CONFIG_IDF_TARGET, info.cores, info.revision / 100, info.revision % 100,
             (unsigned)(flash / (1024 * 1024)),
             (unsigned)(esp_psram_get_size() / (1024 * 1024)));
    ESP_LOGI(TAG, " cpu=%d MHz  idf=%s", (int)esp_clk_cpu_freq(), esp_get_idf_version());
    ESP_LOGI(TAG, " heap internal=%u B psram=%u B",
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    ESP_LOGI(TAG, "==============================================");
}

static void dot_selftest(void)
{
    int8_t *a = heap_caps_aligned_alloc(16, 256, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    int8_t *b = heap_caps_aligned_alloc(16, 256, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!a || !b) {
        ESP_LOGE(TAG, "selftest alloc failed");
        return;
    }
    uint32_t seed = 0x1234abcd;
    for (int i = 0; i < 256; i++) {
        seed = seed * 1103515245u + 12345u;
        a[i] = (int8_t)(seed >> 16);
        seed = seed * 1103515245u + 12345u;
        b[i] = (int8_t)(seed >> 16);
    }
    const int lens[] = {16, 32, 64, 172, 176};
    bool ok = true;
    for (unsigned i = 0; i < sizeof(lens) / sizeof(lens[0]); i++) {
        int32_t v = llm_dot_s8(a, b, lens[i]);
        int32_t s = llm_dot_s8_scalar(a, b, lens[i]);
        ESP_LOGI(TAG, "dot n=%-4d vector=%-8d scalar=%-8d %s",
                 lens[i], (int)v, (int)s, v == s ? "OK" : "MISMATCH");
        if (v != s) {
            ok = false;
        }
    }
    // The fused load+MAC form is only useful if its load target is really
    // independent of the multiply sources, so prove it against the scalar path.
    int32_t fv = llm_dot_s8_fused64(a, b);
    int32_t fs = llm_dot_s8_scalar(a, b, 64);
    ESP_LOGI(TAG, "dot fused64 vector=%-8d scalar=%-8d %s",
             (int)fv, (int)fs, fv == fs ? "OK" : "MISMATCH");
    if (fv != fs) {
        ok = false;
    }

    // Four-position batched K dot. Each lane must match a single dot.
    int32_t q4[4];
    llm_dot16x4_s8(a, b, 32, q4);
    for (int j = 0; j < 4; j++) {
        int32_t s = llm_dot_s8_scalar(a, b + (size_t)j * 32, 16);
        ESP_LOGI(TAG, "dot x4[%d] vector=%-8d scalar=%-8d %s",
                 j, (int)q4[j], (int)s, q4[j] == s ? "OK" : "MISMATCH");
        if (q4[j] != s) {
            ok = false;
        }
    }

    ESP_LOGI(TAG, "SIMD self-test: %s", ok ? "PASS" : "FAIL");
    free(a);
    free(b);
}

// Benchmark for the attention read pattern.
// Attention is only ~1,200 eight-element dots per token, so if it costs 470 us
// then either the FPU is slow or the KV reads stall; measure both.
static void hw_bench(void)
{
    volatile float sink = 0.0f;

    // 3. The attention read pattern: eight floats taken every 128 bytes. Four
    // accumulators, so this measures memory and not FPU dependency latency.
    // The same walk is run from internal SRAM and from PSRAM; the difference
    // is what a PSRAM-resident KV cache actually costs per position.
    float *ps = heap_caps_malloc(512 * 32 * sizeof(float), MALLOC_CAP_SPIRAM);
    float *in = heap_caps_malloc(512 * 32 * sizeof(float), MALLOC_CAP_INTERNAL);
    float *sr = heap_caps_malloc(8 * sizeof(float), MALLOC_CAP_INTERNAL);
    if (ps && in && sr) {
        for (int i = 0; i < 512 * 32; i++) {
            ps[i] = (float)(i & 7);
            in[i] = (float)(i & 7);
        }
        for (int i = 0; i < 8; i++) {
            sr[i] = 0.5f;
        }
        const int REPS = 20;
        float s = 0.0f;
        for (int which = 0; which < 2; which++) {
            const float *base = which ? ps : in;
            int64_t ta = esp_timer_get_time();
            for (int rep = 0; rep < REPS; rep++) {
                for (int t = 0; t < 512; t++) {
                    const float *k = base + (size_t)t * 32;
                    float a0 = k[0] * sr[0], a1 = k[1] * sr[1];
                    float a2 = k[2] * sr[2], a3 = k[3] * sr[3];
                    float b0 = k[4] * sr[4], b1 = k[5] * sr[5];
                    float b2 = k[6] * sr[6], b3 = k[7] * sr[7];
                    s += (a0 + a1) + (a2 + a3) + (b0 + b1) + (b2 + b3);
                }
            }
            int64_t tb = esp_timer_get_time();
            ESP_LOGI(TAG, "bench dot8 stride128 %s: %.1f ns per position",
                     which ? "psram" : "sram ",
                     (double)(tb - ta) * 1000.0 / (REPS * 512.0));
        }
        sink = s;
        free(sr);
        free(in);
        free(ps);
    }

    sink = sink + 1.0f;
    (void)sink;
}

static bool sd_mount(sdmmc_card_t **out_card)
{
    esp_vfs_fat_sdmmc_mount_config_t mount_config = {
        .format_if_mount_failed = false,
        .max_files = 5,
        .allocation_unit_size = 16 * 1024,
    };
    sdmmc_host_t host = SDMMC_HOST_DEFAULT();
    host.max_freq_khz = SDMMC_FREQ_HIGHSPEED;   // 40 MHz; the card reports 40 MHz

    sdmmc_slot_config_t slot = SDMMC_SLOT_CONFIG_DEFAULT();
    slot.width = 1;
    slot.clk = SD_MMC_CLK;
    slot.cmd = SD_MMC_CMD;
    slot.d0 = SD_MMC_D0;
    slot.flags |= SDMMC_SLOT_FLAG_INTERNAL_PULLUP;

    int64_t t0 = esp_timer_get_time();
    esp_err_t err = esp_vfs_fat_sdmmc_mount("/sdcard", &host, &slot, &mount_config, out_card);
    int64_t dt = esp_timer_get_time() - t0;
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "SD mount failed: %s (after %d us)", esp_err_to_name(err), (int)dt);
        return false;
    }
    sdmmc_card_print_info(stdout, *out_card);
    ESP_LOGI(TAG, "SD mounted in %d us", (int)dt);
    return true;
}

// If the filesystem is not FAT (exFAT is disabled in IDF's FATFS build), show
// what is actually on the card so the next step is obvious.
static void sd_raw_probe(void)
{
    sdmmc_host_t host = SDMMC_HOST_DEFAULT();
    host.max_freq_khz = SDMMC_FREQ_HIGHSPEED;   // 40 MHz
    sdmmc_slot_config_t slot = SDMMC_SLOT_CONFIG_DEFAULT();
    slot.width = 1;
    slot.clk = SD_MMC_CLK;
    slot.cmd = SD_MMC_CMD;
    slot.d0 = SD_MMC_D0;
    slot.flags |= SDMMC_SLOT_FLAG_INTERNAL_PULLUP;

    if (sdmmc_host_init() != ESP_OK) {
        ESP_LOGE(TAG, "probe: host init failed");
        return;
    }
    if (sdmmc_host_init_slot(host.slot, &slot) != ESP_OK) {
        ESP_LOGE(TAG, "probe: slot init failed");
        sdmmc_host_deinit();
        return;
    }
    sdmmc_card_t *card = calloc(1, sizeof(sdmmc_card_t));
    if (!card) {
        sdmmc_host_deinit();
        return;
    }
    esp_err_t err = sdmmc_card_init(&host, card);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "probe: card init failed: %s", esp_err_to_name(err));
        free(card);
        sdmmc_host_deinit();
        return;
    }
    sdmmc_card_print_info(stdout, card);

    uint8_t sec[512];
    if (sdmmc_read_sectors(card, sec, 0, 1) == ESP_OK) {
        ESP_LOGI(TAG, "probe: sector0 sig=%02x%02x", sec[510], sec[511]);
        if (memcmp(sec + 3, "EXFAT   ", 8) == 0) {
            ESP_LOGW(TAG, "probe: filesystem = exFAT (volume boot record)");
        } else if (memcmp(sec + 3, "NTFS    ", 8) == 0) {
            ESP_LOGW(TAG, "probe: filesystem = NTFS");
        } else if (memcmp(sec + 82, "FAT32   ", 8) == 0) {
            ESP_LOGI(TAG, "probe: filesystem = FAT32 (volume boot record)");
        } else if (sec[510] == 0x55 && sec[511] == 0xAA) {
            ESP_LOGI(TAG, "probe: MBR/GPT partition table, %02x %02x %02x %02x",
                     sec[446], sec[447], sec[448], sec[449]);
            for (int i = 0; i < 4; i++) {
                const uint8_t *e = sec + 446 + i * 16;
                if (e[4] == 0) {
                    continue;
                }
                uint32_t lba = (uint32_t)e[8] | ((uint32_t)e[9] << 8) |
                               ((uint32_t)e[10] << 16) | ((uint32_t)e[11] << 24);
                uint32_t cnt = (uint32_t)e[12] | ((uint32_t)e[13] << 8) |
                               ((uint32_t)e[14] << 16) | ((uint32_t)e[15] << 24);
                ESP_LOGI(TAG, "probe: part%d type=%02x lba=%u sectors=%u", i, e[4],
                         (unsigned)lba, (unsigned)cnt);
                uint8_t bs[512];
                if (sdmmc_read_sectors(card, bs, lba, 1) == ESP_OK) {
                    if (memcmp(bs + 3, "EXFAT   ", 8) == 0) {
                        ESP_LOGW(TAG, "probe: part%d is exFAT", i);
                    } else if (memcmp(bs + 82, "FAT32   ", 8) == 0) {
                        ESP_LOGI(TAG, "probe: part%d is FAT32", i);
                    } else if (memcmp(bs + 54, "FAT", 3) == 0) {
                        ESP_LOGI(TAG, "probe: part%d is FAT12/16", i);
                    } else {
                        ESP_LOGW(TAG, "probe: part%d unknown boot sector", i);
                    }
                }
            }
        } else {
            ESP_LOGW(TAG, "probe: unrecognised sector 0");
        }
    }
    free(card);
    sdmmc_host_deinit();
}

static void sd_list(const char *path)
{
    DIR *d = opendir(path);
    if (!d) {
        return;
    }
    struct dirent *e;
    int n = 0;
    while ((e = readdir(d)) != NULL && n < 16) {
        struct stat st;
        char full[320];
        snprintf(full, sizeof(full), "%s/%s", path, e->d_name);
        if (stat(full, &st) == 0 && S_ISREG(st.st_mode)) {
            ESP_LOGI(TAG, "  %-28s %8d B", e->d_name, (int)st.st_size);
        } else {
            ESP_LOGI(TAG, "  %-28s <dir>", e->d_name);
        }
        n++;
    }
    closedir(d);
}

static uint8_t *load_sd(const char *path, size_t *out_size)
{
    // read() on the FATFS VFS reaches 4.36 MB/s with 64 KiB requests; the
    // stdio fread() path measured 1.34 MB/s because of its internal buffer.
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        ESP_LOGW(TAG, "open %s failed", path);
        return NULL;
    }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) {
        close(fd);
        return NULL;
    }
    size_t sz = (size_t)st.st_size;
    uint8_t *buf = heap_caps_malloc(sz, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!buf) {
        ESP_LOGE(TAG, "no PSRAM for %u bytes", (unsigned)sz);
        close(fd);
        return NULL;
    }
    size_t off = 0;
    while (off < sz) {
        size_t want = sz - off;
        if (want > 64 * 1024) {
            want = 64 * 1024;
        }
        ssize_t n = read(fd, buf + off, want);
        if (n <= 0) {
            ESP_LOGE(TAG, "short read %u/%u", (unsigned)off, (unsigned)sz);
            free(buf);
            close(fd);
            return NULL;
        }
        off += (size_t)n;
    }
    close(fd);
    *out_size = sz;
    return buf;
}

// Fallback source: a raw flash partition holding uint32 length + file bytes.
static uint8_t *load_part(const char *label, size_t *out_size)
{
    const esp_partition_t *p = esp_partition_find_first(
        ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_ANY, label);
    if (!p) {
        return NULL;
    }
    uint32_t len = 0;
    if (esp_partition_read(p, 0, &len, 4) != ESP_OK || len == 0 || len > p->size) {
        ESP_LOGW(TAG, "partition %s has no valid image", label);
        return NULL;
    }
    uint8_t *buf = heap_caps_malloc(len, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!buf) {
        return NULL;
    }
    if (esp_partition_read(p, 4, buf, len) != ESP_OK) {
        free(buf);
        return NULL;
    }
    *out_size = len;
    ESP_LOGI(TAG, "loaded %s from flash partition (%u B)", label, (unsigned)len);
    return buf;
}

// Stage a model file on the SD card from its flash-partition copy, so the card
// becomes the real model store instead of a fallback that never gets used.
static bool sd_ensure_file(const char *path, const char *label)
{
    struct stat st;
    if (stat(path, &st) == 0 && st.st_size > 0) {
        ESP_LOGI(TAG, "SD already has %s (%d B)", path, (int)st.st_size);
        return true;
    }
    size_t n = 0;
    uint8_t *buf = load_part(label, &n);
    if (!buf) {
        return false;
    }
    // The model partition may hold an .m8 blob while the card is being seeded.
    // Never copy that over the legacy checkpoint path.
    if (n >= 4 && memcmp(buf, "M8B1", 4) == 0) {
        ESP_LOGW(TAG, "partition %s holds an .m8 blob, not a legacy model", label);
        free(buf);
        return false;
    }
    FILE *f = fopen(path, "wb");
    if (!f) {
        ESP_LOGW(TAG, "cannot create %s on the card", path);
        free(buf);
        return false;
    }
    size_t wr = fwrite(buf, 1, n, f);
    fflush(f);
    fclose(f);
    free(buf);
    if (wr != n) {
        ESP_LOGW(TAG, "short write %u/%u for %s", (unsigned)wr, (unsigned)n, path);
        return false;
    }
    ESP_LOGI(TAG, "staged %s on the SD card (%u B)", path, (unsigned)n);
    return true;
}

// One-time card seeding: if /sdcard/models/default.m8 is missing but the flash
// model partition holds a length-prefixed .m8 blob, write it out. This is a
// transport for the first flash of a new blob; the normal boot path only reads.
#if !M8_TEST_PARTITION_ONLY
static void m8_stage_from_partition(void)
{
    struct stat st;
    if (stat(M8_PATH, &st) == 0 && st.st_size > 0) {
        ESP_LOGI(TAG, "%s already on the card (%d B)", M8_PATH, (int)st.st_size);
        return;
    }
    model_stage_t stage;
    int rc = model_stage_partition("model", &stage);
    if (rc != MODEL_OK) {
        ESP_LOGI(TAG, "no .m8 blob in the model partition (%s)",
                 model_strerror(rc));
        return;
    }
    mkdir("/sdcard/models", 0777);
    rc = model_stage_save(&stage, M8_PATH);
    if (rc == MODEL_OK) {
        ESP_LOGI(TAG, "staged %s from the model partition (%u B)",
                 M8_PATH, (unsigned)stage.file_bytes);
    } else {
        ESP_LOGW(TAG, "could not stage %s (%s)", M8_PATH, model_strerror(rc));
    }
    model_stage_free(&stage);
}
#endif

// Boot the model. Order is:
//   1. the model named by the NVS last-selection, if it still loads
//   2. /sdcard/models/default.m8
//   3. the .m8 blob in the flash model partition
//   4. the legacy fp32 checkpoint plus tok512.bin
//
// Ordering matters: every byte is staged into PSRAM while the card is mounted,
// then the card is unmounted before the int8 arena is allocated. The SDMMC and
// FATFS buffers otherwise sit in internal SRAM and push the tail of the weights
// into PSRAM (measured 250,368 B internal with the card mounted vs 260,608 B
// with it unmounted).
//
// Consumes `card`: unmounts it when `sd_ok` is true. On success *out_m and
// *out_tk are live and owned by the caller; legacy mode also leaves the fp32
// checkpoint in m->raw for the reference path. When sw_ctx is non-NULL, the
// committed .m8 stage is moved into it so runtime switching has a rollback
// source.
static bool boot_model(sdmmc_card_t *card, bool sd_ok, int ctx,
                       llm_t **out_m, tokenizer_t *out_tk, bool *out_m8,
                       size_t *out_m8_bytes, model_ctx_t *sw_ctx)
{
    *out_m = NULL;
    *out_m8 = false;
    *out_m8_bytes = 0;
    memset(out_tk, 0, sizeof(*out_tk));

    model_stage_t st;
    memset(&st, 0, sizeof(st));
    int m8rc = MODEL_ERR_IO;
    char used_name[64] = "";
    char last[64] = "";
#if !M8_TEST_PARTITION_ONLY
    if (sd_ok) {
        mkdir(MODEL_DIR, 0777);
        // Seed the card from the flash partition on the first boot after a
        // blob was written there. No-op when the file already exists.
        m8_stage_from_partition();

        if (model_pref_get_last(last, sizeof(last)) == 0) {
            char path[96];
            snprintf(path, sizeof(path), "%s/%s", MODEL_DIR, last);
            m8rc = model_stage_file(path, &st);
            if (m8rc == MODEL_OK) {
                snprintf(used_name, sizeof(used_name), "%s", last);
                ESP_LOGI(TAG, "boot: using NVS last model %s", last);
            } else {
                ESP_LOGW(TAG, "boot: last model %s failed (%s); trying default",
                         last, model_strerror(m8rc));
            }
        }
        if (m8rc != MODEL_OK) {
            m8rc = model_stage_file(M8_PATH, &st);
            if (m8rc == MODEL_OK) {
                snprintf(used_name, sizeof(used_name), "%s", "default.m8");
            }
        }
        // Persist what actually booted. A partition fallback leaves
        // used_name empty and never overwrites the user's selection.
        if (m8rc == MODEL_OK && used_name[0] &&
            strcmp(used_name, last) != 0) {
            model_pref_set_last(used_name);
        }
    }
#endif
    // The partition is a real fallback, not just a transport: with no card, or
    // with a card that has no valid file, boot the blob straight from flash.
    if (m8rc != MODEL_OK) {
        int prc = model_stage_partition("model", &st);
        if (prc == MODEL_OK) {
            ESP_LOGI(TAG, "using the .m8 blob in the model partition");
            m8rc = MODEL_OK;
        } else if (m8rc == MODEL_ERR_IO && prc != MODEL_ERR_IO) {
            m8rc = prc;
        }
    }
    if (m8rc != MODEL_OK && m8rc != MODEL_ERR_IO && m8rc != MODEL_ERR_MAGIC &&
        m8rc != MODEL_ERR_SIZE) {
        ESP_LOGW(TAG, ".m8 rejected (%s); using the legacy checkpoint",
                 model_strerror(m8rc));
    }

    size_t model_size = 0, tok_size = 0;
    uint8_t *model_raw = NULL, *tok_raw = NULL;
    if (m8rc != MODEL_OK) {
        if (sd_ok) {
            sd_ensure_file(MODEL_PATH, "model");
            sd_ensure_file(TOK_PATH, "tok");
            model_raw = load_sd(MODEL_PATH, &model_size);
            tok_raw = load_sd(TOK_PATH, &tok_size);
        }
        if (!model_raw) {
            model_raw = load_part("model", &model_size);
        }
        if (!tok_raw) {
            tok_raw = load_part("tok", &tok_size);
        }
    }

    if (sd_ok) {
        esp_err_t ue = esp_vfs_fat_sdcard_unmount("/sdcard", card);
        ESP_LOGI(TAG, "SD unmounted (%s); internal free=%u largest=%u",
                 esp_err_to_name(ue),
                 (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
                 (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL));
        sd_ok = false;
    }

    llm_t *m = calloc(1, sizeof(llm_t));
    if (!m) {
        model_stage_free(&st);
        free(model_raw);
        free(tok_raw);
        return false;
    }

    if (m8rc == MODEL_OK) {
        size_t bytes = st.file_bytes;
        int rc = model_commit(&st, m, out_tk);
        if (rc == MODEL_OK) {
            if (sw_ctx) {
                if (model_bind(sw_ctx, &st, m, out_tk) != MODEL_OK) {
                    ESP_LOGW(TAG, "switch context bind failed; switching disabled");
                    model_stage_free(&st);
                }
            } else {
                model_stage_free(&st);
            }
            ESP_LOGI(TAG, "model source: .m8 blob");
            *out_m = m;
            *out_m8 = true;
            *out_m8_bytes = bytes;
            return true;
        }
        model_stage_free(&st);
        ESP_LOGW(TAG, ".m8 commit failed (%s); retrying the legacy path",
                 model_strerror(rc));
        // Only reachable if the card has a blob but not a working legacy copy.
        // Remount, read the fallback files, and unmount again before parsing.
        if (!model_raw) {
            sdmmc_card_t *c2 = NULL;
            if (sd_mount(&c2)) {
                sd_ensure_file(MODEL_PATH, "model");
                sd_ensure_file(TOK_PATH, "tok");
                model_raw = load_sd(MODEL_PATH, &model_size);
                tok_raw = load_sd(TOK_PATH, &tok_size);
                if (!model_raw) {
                    model_raw = load_part("model", &model_size);
                }
                if (!tok_raw) {
                    tok_raw = load_part("tok", &tok_size);
                }
                esp_vfs_fat_sdcard_unmount("/sdcard", c2);
            }
        }
    }

    if (!model_raw || !tok_raw) {
        ESP_LOGE(TAG, "no usable model; stopping");
        llm_free_state(m);
        free(m);
        free(model_raw);
        free(tok_raw);
        return false;
    }
    int rc = llm_parse(m, model_raw, model_size, ctx);
    if (rc != 0) {
        ESP_LOGE(TAG, "model parse failed rc=%d", rc);
        llm_free_state(m);
        free(m);
        free(model_raw);
        free(tok_raw);
        return false;
    }
    if (tok_parse(out_tk, tok_raw, tok_size, m->vocab_size) != 0) {
        ESP_LOGE(TAG, "tokenizer parse failed");
        llm_free_state(m);
        free(m);
        free(model_raw);
        free(tok_raw);
        return false;
    }
    ESP_LOGI(TAG, "model source: legacy checkpoint (%u B)", (unsigned)model_size);
    *out_m = m;
    return true;
}

static void run_gen(llm_t *m, tokenizer_t *tk, const char *prompt, int n_gen,
                    bool q8, gen_result_t *r)
{
    int toks[256];
    int np = tok_encode(tk, prompt, 1, toks, 256);
    if (np <= 0) {
        np = 1;
        toks[0] = 1;
    }
    r->n_prompt = np;
    int pos = 0;
    float *lg = m->logits;

    int64_t t0 = esp_timer_get_time();
    for (int i = 0; i < np; i++) {
        if (q8) {
            llm_forward_int8(m, toks[i], pos, lg);
        } else {
            llm_forward_f32(m, toks[i], pos, lg);
        }
        pos++;
    }
    r->ttft_us = esp_timer_get_time() - t0;

    int len = 0;
    r->text[0] = '\0';
    int prev = toks[np - 1];
    for (int g = 0; g < n_gen; g++) {
        int best = 0;
        float bv = lg[0];
        for (int i = 1; i < m->vocab_size; i++) {
            if (lg[i] > bv) {
                bv = lg[i];
                best = i;
            }
        }
        r->ids[g] = best;
        char b[16];
        int n = tok_decode(tk, prev, best, b, sizeof(b));
        if (len + n < (int)sizeof(r->text) - 1) {
            memcpy(r->text + len, b, (size_t)n);
            len += n;
            r->text[len] = '\0';
        }
        prev = best;
        if (pos < m->ctx) {
            if (q8) {
                llm_forward_int8(m, best, pos, lg);
            } else {
                llm_forward_f32(m, best, pos, lg);
            }
            pos++;
        }
    }
    r->n_gen = n_gen;
    r->total_us = esp_timer_get_time() - t0;
}

static void report_gen(const char *label, const gen_result_t *r)
{
    int64_t gen_us = r->total_us - r->ttft_us;
    if (gen_us <= 0) {
        gen_us = 1;
    }
    int64_t tps100 = (int64_t)r->n_gen * 100000000LL / gen_us;
    ESP_LOGI(TAG, "%s: prompt %d tok, ttft %d us, %d gen tok in %d us",
             label, r->n_prompt, (int)r->ttft_us, r->n_gen, (int)gen_us);
    ESP_LOGI(TAG, "%s: %.2f tok/s", label, (double)tps100 / 100.0);
    ESP_LOGI(TAG, "%s: \"%s\"", label, r->text);
    char ids[256];
    int off = 0;
    for (int i = 0; i < r->n_gen && off < (int)sizeof(ids) - 8; i++) {
        off += snprintf(ids + off, sizeof(ids) - (size_t)off, "%s%d", i ? "," : "", r->ids[i]);
    }
    ESP_LOGI(TAG, "%s ids: [%s]", label, ids);
}

// Per-section cost of the int8 forward pass, in microseconds per token.
static void report_prof(const llm_t *m, int forwards)
{
    static const char *names[8] = {
        "qkv matmuls", "rope", "attention", "wo + residual",
        "ffn w1/w3", "silu", "w2 + residual", "final+classifier",
    };
    uint64_t total = 0;
    for (int i = 0; i < 8; i++) {
        total += m->prof[i];
    }
    ESP_LOGI(TAG, "int8 profile: %d forwards, %.2f ms/token accounted",
             forwards, (double)total / 1000.0 / forwards);
    for (int i = 0; i < 8; i++) {
        ESP_LOGI(TAG, "  %-16s %8.1f us/token  %4.1f%%", names[i],
                 (double)m->prof[i] / forwards,
                 total ? 100.0 * (double)m->prof[i] / (double)total : 0.0);
    }
}

static int match_expected(const gen_result_t *r){
    int n = r->n_gen < EXPECTED_N ? r->n_gen : EXPECTED_N;
    for (int i = 0; i < n; i++) {
        if (r->ids[i] != EXPECTED[i]) {
            return i;
        }
    }
    return -1;
}

#if SWITCH_TEST || LIB_TEST
// Test-only provisioning: copy the blob in the flash model partition to
// /sdcard/models/alt.m8. Called with the card mounted, before boot_model().
// Overwrites, so a new test blob always replaces the previous one.
static void switch_stage_partition_to_alt(void)
{
    model_stage_t st;
    int rc = model_stage_partition("model", &st);
    if (rc != MODEL_OK) {
        printf("SWITCH_STAGE rc=%d (%s)\n", rc, model_strerror(rc));
        fflush(stdout);
        return;
    }
    mkdir(MODEL_DIR, 0777);
    rc = model_stage_save(&st, SWITCH_ALT_PATH);
    printf("SWITCH_STAGE rc=%d bytes=%u name=%s\n", rc,
           (unsigned)st.file_bytes, st.name);
    model_stage_free(&st);
    fflush(stdout);
}
#endif

#if SWITCH_TEST
typedef struct {
    uint32_t mount_us;
    uint32_t stage_us;
    uint32_t copy_us;
    uint32_t sanity_us;
    uint32_t total_us;
    bool rolled_back;
} switch_stats_t;

// One full session-change: mount, stage from the card, unmount, apply.
static int switch_model_file(model_ctx_t *ctx, const char *path,
                             switch_stats_t *s)
{
    memset(s, 0, sizeof(*s));
    int64_t t0 = esp_timer_get_time();
    sdmmc_card_t *card = NULL;
    if (!sd_mount(&card)) {
        s->total_us = (uint32_t)(esp_timer_get_time() - t0);
        return MODEL_ERR_IO;
    }
    int64_t t1 = esp_timer_get_time();
    model_stage_t st;
    int rc = model_stage_file(path, &st);
    int64_t t2 = esp_timer_get_time();
    esp_vfs_fat_sdcard_unmount("/sdcard", card);
    s->mount_us = (uint32_t)(t1 - t0);
    s->stage_us = (uint32_t)(t2 - t1);
    if (rc != MODEL_OK) {
        s->total_us = (uint32_t)(esp_timer_get_time() - t0);
        return rc;
    }
    model_apply_stats_t a;
    rc = model_apply(ctx, &st, &a);
    s->copy_us = a.copy_us;
    s->sanity_us = a.sanity_us;
    s->rolled_back = a.rolled_back;
    s->total_us = (uint32_t)(esp_timer_get_time() - t0);
    return rc;
}

static void switch_test(model_ctx_t *ctx)
{
    printf("SWITCH start runs=%d active=%s\n", SWITCH_RUNS, ctx->active.name);
    fflush(stdout);
    int ok = 0, fail = 0, last_rc = MODEL_OK;
    uint32_t min_us = 0xffffffffu, max_us = 0;
    uint64_t sum_us = 0;

    for (int i = 0; i < SWITCH_RUNS; i++) {
        const char *path = (i & 1) ? M8_PATH : SWITCH_ALT_PATH;
        switch_stats_t s;
        last_rc = switch_model_file(ctx, path, &s);
        printf("SWITCH i=%d file=%s rc=%d mount=%u stage=%u copy=%u "
               "sanity=%u total=%u rollback=%d build=%s\n",
               i, path, last_rc, (unsigned)s.mount_us, (unsigned)s.stage_us,
               (unsigned)s.copy_us, (unsigned)s.sanity_us,
               (unsigned)s.total_us, s.rolled_back ? 1 : 0,
               ctx->active.build_id);
        fflush(stdout);
        if (last_rc != MODEL_OK) {
            fail++;
            break;
        }
        ok++;
        if (s.total_us < min_us) min_us = s.total_us;
        if (s.total_us > max_us) max_us = s.total_us;
        sum_us += s.total_us;
    }

    printf("SWITCH_SUM ok=%d fail=%d min_us=%u avg_us=%u max_us=%u\n",
           ok, fail, ok ? (unsigned)min_us : 0,
           ok ? (unsigned)(sum_us / (uint64_t)ok) : 0, (unsigned)max_us);
    printf("SWITCH_ACTIVE name=%s build=%s\n",
           ctx->active.name, ctx->active.build_id);

    // The active model must still be the boot model and still produce the
    // golden ids, whether the last switch succeeded or rolled back.
    gen_result_t g;
    run_gen(ctx->m, ctx->tk, "Once upon a time", GEN_TOKENS, true, &g);
    int match = match_expected(&g) < 0;
    printf("SWITCH_GEN_MATCH=%d\n", match ? 1 : 0);
    if (last_rc == MODEL_ERR_SANITY) {
        printf("SWITCH_ROLLBACK_OK=%d\n", match ? 1 : 0);
    }
    printf("SWITCH_DONE\n");
    fflush(stdout);
}
#endif

#if LIB_TEST
typedef struct {
    bool cache_hit;
    uint32_t mount_us;
    uint32_t stage_us;
    uint32_t copy_us;
    uint32_t sanity_us;
    uint32_t total_us;
} lib_stats_t;

// Application-level selection: PSRAM cache first, then mount + stage + apply.
// The library never mounts the card, so this is the only place that knows
// about /sdcard.
static int lib_select(model_ctx_t *ctx, const char *path, lib_stats_t *s)
{
    memset(s, 0, sizeof(*s));
    int64_t t0 = esp_timer_get_time();

    model_apply_stats_t a;
    int rc = model_apply_cached(ctx, path, &a);
    if (rc != MODEL_ERR_NOTFOUND) {
        s->cache_hit = true;
        s->copy_us = a.copy_us;
        s->sanity_us = a.sanity_us;
        s->total_us = (uint32_t)(esp_timer_get_time() - t0);
        if (rc == MODEL_OK) {
            model_pref_set_last(path);
        }
        return rc;
    }

    sdmmc_card_t *card = NULL;
    int64_t tm = esp_timer_get_time();
    if (!sd_mount(&card)) {
        s->total_us = (uint32_t)(esp_timer_get_time() - t0);
        return MODEL_ERR_IO;
    }
    s->mount_us = (uint32_t)(esp_timer_get_time() - tm);

    int64_t t1 = esp_timer_get_time();
    model_stage_t st;
    rc = model_stage_file(path, &st);
    s->stage_us = (uint32_t)(esp_timer_get_time() - t1);
    esp_vfs_fat_sdcard_unmount("/sdcard", card);
    if (rc != MODEL_OK) {
        s->total_us = (uint32_t)(esp_timer_get_time() - t0);
        return rc;
    }

    rc = model_apply(ctx, &st, &a);
    s->copy_us = a.copy_us;
    s->sanity_us = a.sanity_us;
    s->total_us = (uint32_t)(esp_timer_get_time() - t0);
    if (rc == MODEL_OK) {
        model_pref_set_last(path);
    }
    return rc;
}

static void lib_list(model_ctx_t *ctx)
{
    sdmmc_card_t *card = NULL;
    if (!sd_mount(&card)) {
        printf("LIB_LIST rc=io\n");
        fflush(stdout);
        return;
    }
    model_info_t infos[MODEL_LIST_MAX];
    size_t n = 0;
    int rc = model_scan_dir(MODEL_DIR, infos, MODEL_LIST_MAX, &n);
    printf("LIB_LIST rc=%d n=%u\n", rc, (unsigned)n);
    for (size_t i = 0; i < n; i++) {
        int compat = model_info_compat(ctx, &infos[i]);
        printf("LIB_MODEL file=%s bytes=%u name=\"%s\" sanity=0x%08x "
               "tok=0x%08x compat=%d\n",
               infos[i].file, (unsigned)infos[i].file_bytes, infos[i].name,
               (unsigned)infos[i].sanity_crc, (unsigned)infos[i].tok_crc,
               compat);
    }
    esp_vfs_fat_sdcard_unmount("/sdcard", card);
    fflush(stdout);
}

#if LIB_TEST == 4
// Extra copies of the same validated B blob, used only to fill all three cache
// slots and force LRU eviction. Same payload, distinct cache keys.
static void lib_provision_extra(void)
{
    model_stage_t st;
    int rc = model_stage_partition("model", &st);
    if (rc != MODEL_OK) {
        printf("LIB_PROV rc=%d (%s)\n", rc, model_strerror(rc));
        fflush(stdout);
        return;
    }
    static const char *files[4] = {
        SWITCH_ALT_PATH, "/sdcard/models/alt2.m8",
        "/sdcard/models/alt3.m8", "/sdcard/models/alt4.m8",
    };
    for (int i = 0; i < 4; i++) {
        int r = model_stage_save(&st, files[i]);
        printf("LIB_PROV file=%s bytes=%u rc=%d\n", files[i],
               (unsigned)st.file_bytes, r);
    }
    model_stage_free(&st);
    fflush(stdout);
}
#endif

static void lib_test(model_ctx_t *ctx, int mode)
{
    char last[64] = "";
    model_pref_get_last(last, sizeof(last));
    printf("LIB_BOOT path=%s name=%s last=%s cache=%d psram_free=%u\n",
           ctx->active.path, ctx->active.name, last, model_cache_count(ctx),
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    fflush(stdout);

    if (mode == 2) {
        gen_result_t g;
        run_gen(ctx->m, ctx->tk, "Once upon a time", GEN_TOKENS, true, &g);
        printf("LIB_GEN match=%d\n", match_expected(&g) < 0 ? 1 : 0);
        printf("LIB_ACTIVE name=%s build=%s path=%s cache=%d\n",
               ctx->active.name, ctx->active.build_id, ctx->active.path,
               model_cache_count(ctx));
        printf("LIB_DONE\n");
        fflush(stdout);
        return;
    }

    if (mode == 3) {
        // Select default.m8, persist it, reboot. The next boot re-runs this
        // mode and hits the confirm branch below.
        if (strcmp(ctx->active.path, M8_PATH) == 0 &&
            strcmp(last, "default.m8") == 0) {
            printf("LIB_BOOT_CONFIRM default.m8\n");
            printf("LIB_ACTIVE name=%s build=%s path=%s cache=%d\n",
                   ctx->active.name, ctx->active.build_id, ctx->active.path,
                   model_cache_count(ctx));
            printf("LIB_DONE\n");
            fflush(stdout);
            return;
        }
        lib_stats_t s;
        int rc = lib_select(ctx, M8_PATH, &s);
        printf("LIB_SELECT i=0 file=%s src=%s rc=%d mount=%u stage=%u "
               "copy=%u sanity=%u total=%u cache=%d active=%s\n",
               M8_PATH, s.cache_hit ? "cache" : "disk", rc,
               (unsigned)s.mount_us, (unsigned)s.stage_us,
               (unsigned)s.copy_us, (unsigned)s.sanity_us,
               (unsigned)s.total_us, model_cache_count(ctx),
               ctx->active.name);
        if (rc == MODEL_OK) {
            printf("LIB_RESTART\n");
            fflush(stdout);
            esp_restart();
        }
        printf("LIB_DONE\n");
        fflush(stdout);
        return;
    }

    if (mode == 4) {
        lib_list(ctx);
        static const char *seq[8] = {
            SWITCH_ALT_PATH, "/sdcard/models/alt2.m8",
            "/sdcard/models/alt3.m8", "/sdcard/models/alt4.m8",
            M8_PATH, "/sdcard/models/alt3.m8",
            "/sdcard/models/alt2.m8", SWITCH_ALT_PATH,
        };
        int ok = 0;
        for (int i = 0; i < 8; i++) {
            lib_stats_t s;
            int rc = lib_select(ctx, seq[i], &s);
            printf("LIB_SELECT i=%d file=%s src=%s rc=%d total=%u cache=%d "
                   "active=%s\n",
                   i, seq[i], s.cache_hit ? "cache" : "disk", rc,
                   (unsigned)s.total_us, model_cache_count(ctx),
                   ctx->active.name);
            fflush(stdout);
            if (rc != MODEL_OK) {
                break;
            }
            ok++;
        }
        // Leave the board on default.m8.
        lib_stats_t s;
        int rc = lib_select(ctx, M8_PATH, &s);
        printf("LIB_FINAL file=%s src=%s rc=%d total=%u cache=%d active=%s\n",
               M8_PATH, s.cache_hit ? "cache" : "disk", rc,
               (unsigned)s.total_us, model_cache_count(ctx), ctx->active.name);
        printf("LIB_SUM ok=%d/8 cache=%d psram_free=%u\n", ok,
               model_cache_count(ctx),
               (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
        printf("LIB_ACTIVE name=%s build=%s path=%s cache=%d\n",
               ctx->active.name, ctx->active.build_id, ctx->active.path,
               model_cache_count(ctx));
        sdmmc_card_t *card = NULL;
        if (sd_mount(&card)) {
            for (int i = 1; i < 4; i++) {
                char path[64];
                snprintf(path, sizeof(path), "%s/alt%d.m8", MODEL_DIR, i + 1);
                unlink(path);
            }
            unlink(SWITCH_ALT_PATH);
            esp_vfs_fat_sdcard_unmount("/sdcard", card);
        }
        printf("LIB_DONE\n");
        fflush(stdout);
        return;
    }

    lib_list(ctx);

    static const char *seq[3] = { SWITCH_ALT_PATH, M8_PATH, SWITCH_ALT_PATH };
    int ok = 0;
    for (int i = 0; i < 3; i++) {
        lib_stats_t s;
        int rc = lib_select(ctx, seq[i], &s);
        printf("LIB_SELECT i=%d file=%s src=%s rc=%d mount=%u stage=%u "
               "copy=%u sanity=%u total=%u cache=%d active=%s\n",
               i, seq[i], s.cache_hit ? "cache" : "disk", rc,
               (unsigned)s.mount_us, (unsigned)s.stage_us,
               (unsigned)s.copy_us, (unsigned)s.sanity_us,
               (unsigned)s.total_us, model_cache_count(ctx), ctx->active.name);
        fflush(stdout);
        if (rc != MODEL_OK) {
            break;
        }
        ok++;
    }

    gen_result_t g;
    run_gen(ctx->m, ctx->tk, "Once upon a time", GEN_TOKENS, true, &g);
    printf("LIB_GEN match=%d\n", match_expected(&g) < 0 ? 1 : 0);
    printf("LIB_SUM ok=%d/%d cache=%d psram_free=%u\n", ok, 3,
           model_cache_count(ctx),
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    printf("LIB_ACTIVE name=%s build=%s path=%s cache=%d\n",
           ctx->active.name, ctx->active.build_id, ctx->active.path,
           model_cache_count(ctx));
    printf("LIB_DONE\n");
    fflush(stdout);
}
#endif

static volatile uint32_t g_bench_sink;

static int64_t bench_cpu(const char *label, uint32_t iters)
{
    int64_t t0 = esp_timer_get_time();
    uint32_t a = 0x12345678u;
    for (uint32_t i = 0; i < iters; i++) {
        a = a * 1664525u + 1013904223u;
        a ^= a >> 13;
    }
    g_bench_sink = a;
    int64_t dt = esp_timer_get_time() - t0;
    ESP_LOGI(TAG, "cpu bench %s: %u iters in %d us (sum %08x)", label, (unsigned)iters,
             (int)dt, (unsigned)a);
    return dt;
}

// 320 MHz is not a Kconfig option: clk_ll_cpu_set_freq_mhz_from_pll() aborts on
// anything other than 80/160/240. The hardware can still do it: on a 320 MHz
// PLL, CPUPERIOD_SEL=2 means PLL/1 = 320 MHz instead of PLL/2 = 240 MHz.
//
// Doing it via rtc_clk_cpu_freq_set_config() alone hangs, because that path
// parks the CPU on XTAL first, which powers the BBPLL down; the re-enable then
// never finishes analog calibration. Instead we stay on the PLL the whole time
// and pivot through 80 MHz, which is PLL/6 at 480 MHz and PLL/4 at 320 MHz.
// The CPU and APB stay at 80 MHz across the PLL reprogram, so the console UART
// keeps its baud rate and every step below stays readable.
static bool bbpll_program(uint32_t mhz, int timeout_ms)
{
    clk_ll_bbpll_set_freq_mhz(mhz);
    regi2c_ctrl_ll_bbpll_calibration_start();
    clk_ll_bbpll_set_config(mhz, 40);
    int64_t t0 = esp_timer_get_time();
    while (!regi2c_ctrl_ll_bbpll_calibration_is_done()) {
        if (esp_timer_get_time() - t0 > (int64_t)timeout_ms * 1000) {
            regi2c_ctrl_ll_bbpll_calibration_stop();
            ESP_LOGE(TAG, "bbpll %u MHz calibration timeout after %d ms",
                     (unsigned)mhz, timeout_ms);
            return false;
        }
    }
    esp_rom_delay_us(10);
    regi2c_ctrl_ll_bbpll_calibration_stop();
    ESP_LOGI(TAG, "bbpll -> %u MHz, calibration done in %d us",
             (unsigned)mhz, (int)(esp_timer_get_time() - t0));
    return true;
}

static void pll_switch(uint32_t src_mhz, uint32_t freq_mhz, uint32_t div)
{
    rtc_cpu_freq_config_t cfg = {
        .source = SOC_CPU_CLK_SRC_PLL,
        .source_freq_mhz = src_mhz,
        .freq_mhz = freq_mhz,
        .div = div,
    };
    rtc_clk_cpu_freq_set_config(&cfg);
}

static void cpu_set_240(void)
{
    // Same pivot in reverse. get_config() cannot describe "320 PLL + SEL2", so
    // it logs and leaves the struct untouched here; the worst case is a brief
    // over-voltage, never an under-voltage.
    pll_switch(320, 80, 4);
    if (!bbpll_program(480, 200)) {
        ESP_LOGE(TAG, "restore: PLL calibration failed, rebooting");
        esp_restart();
    }
    pll_switch(480, 240, 2);
}

#if BENCH_MODE

// ---- paper measurement mode -------------------------------------------------
// One boot runs: optional logit dump, VAR_RUNS short generations, then CTX_RUNS
// full-context generations. Timings are collected into arrays and printed after
// the run so UART output never lands inside a measured region.

static uint32_t s_tim_us[CTX_LEN + 64];
static int s_tim_pos[CTX_LEN + 64];
static int s_tok_id[CTX_LEN + 64];

static void bench_dump_logits(const float *lg, int n, const char *label, int step)
{
    printf("LOGD %s %d %d\n", label, step, n);
    for (int i = 0; i < n; i++) {
        union { float f; uint32_t u; } v;
        v.f = lg[i];
        printf("%08x%c", (unsigned)v.u, (i % 16 == 15) ? '\n' : ' ');
        if ((i & 15) == 15) {
            // The UART write blocks; yield so the idle task can feed the WDT.
            fflush(stdout);
            vTaskDelay(1);
        }
    }
    printf("LOGE %s %d\n", label, step);
    fflush(stdout);
}

#if KV_INT8 || KVCHECK
// Long-context accuracy check. A 124-token prompt is repeated to fill the
// 512-token context, then fp32 and int8 logits are dumped at several depths.
// This covers the hot-window boundary (15/16/17) and the deep int8 regime.
static void bench_kvcheck(llm_t *m, tokenizer_t *tk)
{
    static const char *para =
        "Once upon a time, in a small village by the sea, there lived a "
        "curious young girl who loved to read stories about distant lands "
        "and the people who lived there. Every evening she would sit by the "
        "window and imagine the ships and the mountains and the cities beyond "
        "the horizon, and she would write down everything she saw in her "
        "little notebook. One day a traveler came to the village and told her "
        "about a library that held a book for every question, and she decided "
        "that she would find it before the winter came.";
    static const int targets[] = { 15, 16, 17, 32, 64, 128, 256, 384, 511 };
    const int nt = (int)(sizeof(targets) / sizeof(targets[0]));
    const int n = m->ctx;
    int toks[512];

    int np = tok_encode(tk, para, 1, toks, 512);
    if (np <= 0) {
        return;
    }
    for (int i = np; i < n; i++) {
        toks[i] = toks[i % np];
    }

    for (int i = 0; i < n; i++) {
        if (llm_forward_f32(m, toks[i], i, m->logits) != 0) {
            return;
        }
        for (int k = 0; k < nt; k++) {
            if (i == targets[k]) {
                char label[16];
                snprintf(label, sizeof(label), "fp32L%d", i);
                bench_dump_logits(m->logits, m->vocab_size, label, 0);
            }
        }
    }

    for (int i = 0; i < n; i++) {
        if (llm_forward_int8(m, toks[i], i, m->logits) != 0) {
            return;
        }
        for (int k = 0; k < nt; k++) {
            if (i == targets[k]) {
                char label[16];
                snprintf(label, sizeof(label), "int8L%d", i);
                bench_dump_logits(m->logits, m->vocab_size, label, 0);
            }
        }
    }
    printf("KVCHECK n=%d targets=%d\n", n, nt);
    fflush(stdout);
}
#endif

static void bench_run(llm_t *m, tokenizer_t *tk, const char *prompt, int n_gen,
                      bool q8, int rep, const char *label, bool dump)
{
    int toks[256];
    int np = tok_encode(tk, prompt, 1, toks, 256);
    if (np <= 0) {
        np = 1;
        toks[0] = 1;
    }

    float *lg = m->logits;
    int pos = 0;
    int ntim = 0;
    int64_t gen_us = 0;
    int64_t t_start = esp_timer_get_time();

    for (int i = 0; i < np; i++) {
        int64_t a = esp_timer_get_time();
        int rc = q8 ? llm_forward_int8(m, toks[i], pos, lg)
                    : llm_forward_f32(m, toks[i], pos, lg);
        int64_t b = esp_timer_get_time();
        if (rc != 0) {
            printf("ERR forward rc=%d pos=%d\n", rc, pos);
            return;
        }
        s_tim_us[ntim] = (uint32_t)(b - a);
        s_tim_pos[ntim] = pos;
        ntim++;
        pos++;
    }
    int64_t ttft_us = esp_timer_get_time() - t_start;

    int ntok = 0;
    for (int g = 0; g < n_gen; g++) {
        if (dump) {
            bench_dump_logits(lg, m->vocab_size, label, g);
        }
        int best = 0;
        float bv = lg[0];
        for (int i = 1; i < m->vocab_size; i++) {
            if (lg[i] > bv) {
                bv = lg[i];
                best = i;
            }
        }
        s_tok_id[ntok++] = best;
        if (pos < m->ctx) {
            int64_t a = esp_timer_get_time();
            int rc = q8 ? llm_forward_int8(m, best, pos, lg)
                        : llm_forward_f32(m, best, pos, lg);
            int64_t b = esp_timer_get_time();
            if (rc != 0) {
                printf("ERR forward rc=%d pos=%d\n", rc, pos);
                return;
            }
            if (pos >= np) {
                gen_us += (b - a);
            }
            s_tim_us[ntim] = (uint32_t)(b - a);
            s_tim_pos[ntim] = pos;
            ntim++;
            pos++;
        }
    }
    int64_t total = esp_timer_get_time() - t_start;

    printf("TTFT %s %d %d %lld\n", label, rep, np, (long long)ttft_us);
    printf("RUN %s %d %d %d %lld %lld\n", label, rep, np, ntok,
           (long long)total, (long long)gen_us);
    for (int i = 0; i < ntim; i++) {
        printf("TIM %s %d %d %u\n", label, rep, s_tim_pos[i], (unsigned)s_tim_us[i]);
        if ((i & 31) == 31) {
            fflush(stdout);
            vTaskDelay(1);
        }
    }
    for (int i = 0; i < ntok; i++) {
        printf("TOK %s %d %d %d\n", label, rep, i, s_tok_id[i]);
        if ((i & 31) == 31) {
            fflush(stdout);
            vTaskDelay(1);
        }
    }
    fflush(stdout);
}

static void bench_mem(const llm_t *m, const char *label)
{
    const qmat_t *all[] = { &m->tok_emb, &m->wq, &m->wk, &m->wv,
                            &m->wo, &m->w1, &m->w2, &m->w3 };
    size_t win = 0, wps = 0, scales = 0;
    for (int i = 0; i < 8; i++) {
        size_t n = (size_t)all[i]->rows * all[i]->stride;
        if (esp_ptr_internal(all[i]->q)) {
            win += n;
        } else {
            wps += n;
        }
        scales += (size_t)all[i]->rows * sizeof(float);
    }
    size_t hot_bytes = 2 * (size_t)m->n_layers * m->n_kv_heads * m->hot *
                       m->head_size * sizeof(float);
    size_t kv = 2 * (size_t)m->n_layers * m->ctx * m->kv_dim * sizeof(float);
    printf("MEM %s hot=%d hot_bytes=%u weights_internal=%u weights_psram=%u "
           "scales=%u free_internal=%u largest=%u free_psram=%u kv=%u model=%u\n",
           label, m->hot, (unsigned)hot_bytes, (unsigned)win, (unsigned)wps,
           (unsigned)scales,
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
           (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL),
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
           (unsigned)kv, (unsigned)m->raw_size);
    fflush(stdout);
}

// KV footprint and the PSRAM reclaimed when the fp32 caches are freed. The
// byte counts are computed from the model dimensions; `reclaimed` is measured
// by the allocator, so it also captures any bookkeeping overhead.
static void bench_kvmem(llm_t *m, const char *label)
{
#if KV_INT8
    size_t n = (size_t)m->n_layers * m->ctx * m->kv_dim;
    size_t fp32_bytes = 2 * n * sizeof(float);
    size_t data = 2 * n;
    size_t scales = 2 * (size_t)m->n_layers * m->ctx * m->n_kv_heads * sizeof(float);
    size_t before = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    llm_drop_fp32_cache(m);
    size_t after = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    printf("KVMEM %s fp32_bytes=%u int8_data=%u int8_scales=%u int8_total=%u "
           "ratio=%.2f reclaimed=%u free_psram=%u\n",
           label, (unsigned)fp32_bytes, (unsigned)data, (unsigned)scales,
           (unsigned)(data + scales), (double)fp32_bytes / (double)(data + scales),
           (unsigned)(after - before), (unsigned)after);
    fflush(stdout);
#else
    (void)m;
    (void)label;
#endif
}

// Raw memory bandwidth probe. The pointers are volatile so the compiler
// cannot vectorise or drop the loads. MB/s = bytes / microseconds.
static uint32_t bw_read_seq(volatile uint32_t *p, size_t words)
{
    uint32_t s = 0;
    for (size_t i = 0; i < words; i++) {
        s += p[i];
    }
    return s;
}

// 16 useful bytes out of every 32, the K dot access pattern.
static uint32_t bw_read_stride(volatile uint32_t *p, size_t words)
{
    uint32_t s = 0;
    for (size_t i = 0; i + 7 < words; i += 8) {
        s += p[i] + p[i + 1] + p[i + 2] + p[i + 3];
    }
    return s;
}

static uint32_t bw_write_seq(volatile uint32_t *p, size_t words)
{
    for (size_t i = 0; i < words; i++) {
        p[i] = (uint32_t)i;
    }
    return 0;
}

static void bench_psram(void)
{
    const size_t ext_bytes = 512 * 1024;
    volatile uint32_t *ext = heap_caps_malloc(ext_bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!ext) {
        printf("PSRAM bench skipped\n");
        return;
    }
    size_t ew = ext_bytes / 4;
    uint32_t sink = 0;
    int64_t best;

    // Warm the mapping.
    sink += bw_read_seq(ext, ew);

    best = 1 << 30;
    for (int r = 0; r < 3; r++) {
        int64_t a = esp_timer_get_time();
        sink += bw_read_seq(ext, ew);
        int64_t b = esp_timer_get_time();
        if (b - a < best) best = b - a;
    }
    double ext_seq = (double)ext_bytes / (double)best;

    best = 1 << 30;
    for (int r = 0; r < 3; r++) {
        int64_t a = esp_timer_get_time();
        sink += bw_read_stride(ext, ew);
        int64_t b = esp_timer_get_time();
        if (b - a < best) best = b - a;
    }
    double ext_str = ((double)ew / 8.0 * 16.0) / (double)best;

    best = 1 << 30;
    for (int r = 0; r < 3; r++) {
        int64_t a = esp_timer_get_time();
        sink += bw_write_seq(ext, ew);
        int64_t b = esp_timer_get_time();
        if (b - a < best) best = b - a;
    }
    double ext_wr = (double)ext_bytes / (double)best;

    printf("PSRAM ext_seq_mbps=%.1f ext_stride_mbps=%.1f ext_write_mbps=%.1f sink=%u\n",
           ext_seq, ext_str, ext_wr, (unsigned)sink);
    fflush(stdout);
    free((void *)ext);
}

static void bench_main(void)
{
    // Long UART prints (512 logits x 12 steps, 512 timing lines) block the
    // calling task; the idle-task watchdog is not useful during a measurement
    // run and its backtraces would corrupt the machine-readable output.
    esp_task_wdt_deinit();
    banner();
    dot_selftest();
    bench_psram();

    sdmmc_card_t *card = NULL;
    bool sd_ok = sd_mount(&card);

#if SWITCH_TEST || LIB_TEST
    // Provision the second model from the partition while the card is mounted.
    if (sd_ok) {
        switch_stage_partition_to_alt();
    }
#endif
#if LIB_TEST == 4
    if (sd_ok) {
        lib_provision_extra();
    }
#endif
#if LIB_TEST == 1
    // Mode 1 starts from default.m8 regardless of what a previous run stored.
    model_pref_clear();
#endif
#if SWITCH_TEST || LIB_TEST
    model_ctx_t *sw_ctx = &g_model;
#else
    model_ctx_t *sw_ctx = NULL;
#endif

    llm_t *m = NULL;
    tokenizer_t tk;
    bool m8 = false;
    size_t m8_bytes = 0;
    if (!boot_model(card, sd_ok, 0, &m, &tk, &m8, &m8_bytes, sw_ctx)) {
        printf("FATAL no model\n");
        return;
    }
    printf("CFG dim=%d hidden=%d layers=%d heads=%d kv_heads=%d vocab=%d "
           "seq=%d ctx=%d hot=%d psram_weights=%d m8=%d\n",
           m->dim, m->hidden_dim, m->n_layers, m->n_heads, m->n_kv_heads,
           m->vocab_size, m->seq_len, m->ctx, m->hot, WEIGHTS_PSRAM, m8 ? 1 : 0);
    if (m8) {
        printf("M8 bytes=%u prequantized\n", (unsigned)m8_bytes);
    }

#if LIB_TEST
    if (!m8 || !g_model.bound) {
        printf("LIB_TEST requires an .m8 boot with a bound context\n");
        printf("LIB_DONE\n");
        fflush(stdout);
        return;
    }
    lib_test(&g_model, LIB_TEST);
    printf("BENCH_DONE\n");
    fflush(stdout);
    return;
#endif

#if SWITCH_TEST
    if (!m8 || !g_model.bound) {
        printf("SWITCH requires an .m8 boot with a bound context\n");
        printf("SWITCH_DONE\n");
        fflush(stdout);
        return;
    }
    switch_test(&g_model);
    printf("BENCH_DONE\n");
    fflush(stdout);
    return;
#endif

    if (!m8) {
        int64_t tq = esp_timer_get_time();
        if (llm_quantize(m, !WEIGHTS_PSRAM) != 0) {
            printf("FATAL quantize\n");
            return;
        }
        printf("QUANT us=%lld\n", (long long)(esp_timer_get_time() - tq));
    } else {
        printf("QUANT skipped (m8 prequantized)\n");
    }
    bench_mem(m, "base");

    const char *prompt = "Once upon a time";
    int ptoks[32];
    int np = tok_encode(&tk, prompt, 1, ptoks, 32);
    printf("PROMPT n=%d ids=", np);
    for (int i = 0; i < np; i++) {
        printf("%s%d", i ? "," : "", ptoks[i]);
    }
    printf("\n");

#if DUMP_LOGITS
    if (!m8) {
        // The fp32 path needs the legacy fp32 checkpoint, which an .m8 boot
        // does not load. The int8 dump below is the one compared to golden.
        bench_run(m, &tk, prompt, VAR_TOKENS, false, 0, "fp32", true);
    }
    bench_run(m, &tk, prompt, VAR_TOKENS, true, 0, "int8", true);
#endif

#if KVCHECK
    // bench_kvcheck drives llm_forward_f32, so it only applies to the legacy
    // checkpoint. The int8 KV correctness coverage comes from the golden dump.
    if (!m8) {
        bench_kvcheck(m, &tk);
    }
#endif
#if KV_INT8
    bench_kvmem(m, "int8only");
#endif

    for (int r = 0; r < VAR_RUNS; r++) {
        bench_run(m, &tk, prompt, VAR_TOKENS, true, r, "var", false);
    }

    // Coarse section profile over the full-context runs. Each run does exactly
    // m->ctx forwards because the prompt tokens and the generated tokens share
    // the same position budget.
    memset(m->prof, 0, sizeof(m->prof));
    memset(m->aprof, 0, sizeof(m->aprof));
    memset(m->aprof2, 0, sizeof(m->aprof2));
    for (int r = 0; r < CTX_RUNS; r++) {
        bench_run(m, &tk, prompt, CTX_LEN, true, r, "ctx", false);
    }
    {
        static const char *names[8] = {
            "qkv", "rope", "attn", "wo", "ffn1", "silu", "ffn2", "head",
        };
        int forwards = CTX_RUNS * m->ctx;
        uint64_t total = 0;
        for (int i = 0; i < 8; i++) {
            total += m->prof[i];
        }
        printf("PROF forwards=%d total_us=%lld\n", forwards, (long long)total);
        for (int i = 0; i < 8; i++) {
            printf("PROF %-6s us_per_tok=%.2f pct=%.1f\n", names[i],
                   (double)m->prof[i] / forwards,
                   total ? 100.0 * (double)m->prof[i] / (double)total : 0.0);
        }

        // Attention sub-profile, both cores summed. Cycles are converted at
        // the 240 MHz CPU clock.
        static const char *anames[4] = { "kdot", "softmax", "vquant", "vdot" };
        uint64_t atotal = 0;
        for (int i = 0; i < 4; i++) {
            atotal += m->aprof[i] + m->aprof2[i];
        }
        printf("APROF forwards=%d cycles=%lld\n", forwards, (long long)atotal);
        for (int i = 0; i < 4; i++) {
            uint64_t c = m->aprof[i] + m->aprof2[i];
            printf("APROF %-8s us_per_tok=%.2f pct=%.1f\n", anames[i],
                   (double)c / forwards / 240.0,
                   atotal ? 100.0 * (double)c / (double)atotal : 0.0);
        }
        fflush(stdout);
    }

    bench_mem(m, "final");
    printf("BENCH_DONE\n");
    fflush(stdout);
}

#endif

void app_main(void)
{
#if BENCH_MODE
    bench_main();
    return;
#endif
    banner();
    dot_selftest();
    hw_bench();

    sdmmc_card_t *card = NULL;
    bool sd_ok = sd_mount(&card);
    if (sd_ok) {
        ESP_LOGI(TAG, "SD root:");
        sd_list("/sdcard");
    } else {
        ESP_LOGW(TAG, "mount failed, probing the raw card");
        sd_raw_probe();
    }

    size_t m8_bytes = 0;
    bool m8 = false;
    llm_t *m = NULL;
    tokenizer_t tk;
    if (!boot_model(card, sd_ok, CTX_TOKENS, &m, &tk, &m8, &m8_bytes, &g_model)) {
        ESP_LOGE(TAG, "no model available; stopping");
        return;
    }
    ESP_LOGI(TAG, "model: dim=%d hidden=%d layers=%d heads=%d kv_heads=%d vocab=%d seq=%d ctx=%d m8=%d",
             m->dim, m->hidden_dim, m->n_layers, m->n_heads, m->n_kv_heads,
             m->vocab_size, m->seq_len, m->ctx, m8 ? 1 : 0);
    if (m8) {
        ESP_LOGI(TAG, "source: .m8 blob (%u B), tokenizer embedded", (unsigned)m8_bytes);
    }
    ESP_LOGI(TAG, "tokenizer: vocab=%d max_token_len=%d", tk.vocab_size, tk.max_token_length);

    const char *prompt = "Once upon a time";
    int probe[32];
    int np = tok_encode(&tk, prompt, 1, probe, 32);
    char pline[128];
    int off = 0;
    for (int i = 0; i < np && off < (int)sizeof(pline) - 8; i++) {
        off += snprintf(pline + off, sizeof(pline) - (size_t)off, "%s%d", i ? "," : "", probe[i]);
    }
    ESP_LOGI(TAG, "prompt \"%s\" -> [%s]", prompt, pline);

    // Quantize before the fp32 run so the int8 weights get first pick of the
    // internal heap, before anything else fragments it. An .m8 boot already has
    // prequantized weights in the arena.
    if (!m8) {
        int64_t tq = esp_timer_get_time();
        if (llm_quantize(m, true) != 0) {
            ESP_LOGE(TAG, "quantize failed");
            return;
        }
        ESP_LOGI(TAG, "int8 quantize took %d us", (int)(esp_timer_get_time() - tq));
    } else {
        ESP_LOGI(TAG, "int8 weights came from the .m8 blob");
    }
    llm_print_mem(m);

    // ---- fp32 reference path (correctness) ----
    // Only the legacy checkpoint carries the fp32 tensors. The .m8 build is
    // checked against the golden int8 logits instead.
    if (!m8) {
        gen_result_t gf;
        int64_t t0 = esp_timer_get_time();
        run_gen(m, &tk, prompt, GEN_TOKENS, false, &gf);
        ESP_LOGI(TAG, "fp32 first pass done in %d us", (int)(esp_timer_get_time() - t0));
        report_gen("fp32", &gf);
        int bad = match_expected(&gf);
        ESP_LOGI(TAG, "fp32 vs reference: %s", bad < 0 ? "MATCH" : "DIFFERS");
    }

    // ---- int8 SIMD path ----
    memset(m->prof, 0, sizeof(m->prof));
    gen_result_t gq;
    run_gen(m, &tk, prompt, GEN_TOKENS, true, &gq);
    report_gen("int8", &gq);
    int bad8 = match_expected(&gq);
    ESP_LOGI(TAG, "int8 vs fp32 reference: %s", bad8 < 0 ? "MATCH" : "DIFFERS");
    report_prof(m, gq.n_prompt + gq.n_gen);
    if (bad8 >= 0) {
        ESP_LOGI(TAG, "  first differing token index %d (int8=%d expected=%d)",
                 bad8, gq.ids[bad8], EXPECTED[bad8]);
    }

    // ---- CPU clock baseline ----
    int64_t base = bench_cpu("240MHz", 3000000);

    // ---- clock experiment: what does a 320 MHz PLL actually give us? ----
    // Measured result from the previous run: SEL=2 on a 320 MHz PLL is PLL/2 =
    // 160 MHz, i.e. 1.50x the 240 MHz time. Generation then hangs, so the PSRAM
    // path is not usable in that state. The only untried selector is SEL=3.
    ESP_LOGW(TAG, "clock experiment: 320 MHz PLL");
    int64_t t_oc = esp_timer_get_time();
    pll_switch(480, 80, 6);
    if (bbpll_program(320, 200)) {
        pll_switch(320, 240, 1);                    // SEL=2
        int64_t t2 = bench_cpu("320PLL/sel2", 3000000);
        ESP_LOGI(TAG, "sel2: %d%% of the 240 MHz time", (int)(t2 * 100 / base));

        ESP_LOGW(TAG, "probing SEL=3 on the same PLL");
        REG_SET_FIELD(SYSTEM_CPU_PER_CONF_REG, SYSTEM_CPUPERIOD_SEL, 3);
        int64_t t3 = bench_cpu("320PLL/sel3", 3000000);
        ESP_LOGI(TAG, "sel3: %d%% of the 240 MHz time", (int)(t3 * 100 / base));

        REG_SET_FIELD(SYSTEM_CPU_PER_CONF_REG, SYSTEM_CPUPERIOD_SEL, 0);
    }
    ESP_LOGW(TAG, "clock experiments done in %d us", (int)(esp_timer_get_time() - t_oc));

    cpu_set_240();
    int64_t back = bench_cpu("240MHz again", 3000000);
    ESP_LOGI(TAG, "restore check: %d%% of the original 240 MHz time", (int)(back * 100 / base));

    // ---- soak at 240 MHz: identical output every run means no corruption ----
    bool stable = true;
    for (int i = 0; i < SOAK_RUNS; i++) {
        gen_result_t g;
        run_gen(m, &tk, prompt, GEN_TOKENS, true, &g);
        bool same = g.n_gen == gq.n_gen;
        for (int j = 0; same && j < g.n_gen; j++) {
            if (g.ids[j] != gq.ids[j]) {
                same = false;
            }
        }
        ESP_LOGI(TAG, "soak %d/%d: %s", i + 1, SOAK_RUNS, same ? "identical" : "DIVERGED");
        if (!same) {
            stable = false;
        }
    }
    ESP_LOGI(TAG, "soak: %s", stable ? "PASS" : "FAIL");

    ESP_LOGI(TAG, "final clock: %d Hz", (int)clk_hal_cpu_get_freq_hz());
    ESP_LOGI(TAG, "done; free internal=%u psram=%u",
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
}
