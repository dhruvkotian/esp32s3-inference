// NVS-backed "last selected model" preference.
//
// Kept in its own translation unit with no dependency on model_lib.c so the
// library stays filesystem/flash agnostic. Init/deinit per call is deliberate:
// holding an NVS handle keeps a page cache in internal SRAM, and on this board
// the int8 weights can otherwise spill into PSRAM.

#include "model_prefs.h"

#include <stdbool.h>
#include <string.h>

#include "esp_log.h"
#include "nvs.h"
#include "nvs_flash.h"

static const char *TAG = "modelpref";

#define PREF_NAMESPACE "modellib"
#define PREF_KEY       "last"

static bool prefs_open(nvs_handle_t *h, nvs_open_mode_t mode)
{
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES ||
        err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        // First use on a blank or older partition layout.
        ESP_LOGW(TAG, "NVS needs an erase (%s)", esp_err_to_name(err));
        nvs_flash_erase();
        err = nvs_flash_init();
    }
    if (err != ESP_OK) {
        ESP_LOGW(TAG, "nvs_flash_init: %s", esp_err_to_name(err));
        return false;
    }
    err = nvs_open(PREF_NAMESPACE, mode, h);
    if (err != ESP_OK) {
        ESP_LOGW(TAG, "nvs_open: %s", esp_err_to_name(err));
        nvs_flash_deinit();
        return false;
    }
    return true;
}

static void prefs_close(nvs_handle_t h)
{
    nvs_close(h);
    nvs_flash_deinit();
}

int model_pref_get_last(char *out, size_t cap)
{
    if (!out || cap < 2) {
        return -1;
    }
    out[0] = '\0';
    nvs_handle_t h;
    if (!prefs_open(&h, NVS_READONLY)) {
        return -1;
    }
    size_t len = cap;
    esp_err_t err = nvs_get_str(h, PREF_KEY, out, &len);
    prefs_close(h);
    if (err != ESP_OK || out[0] == '\0') {
        out[0] = '\0';
        return -1;
    }
    return 0;
}

int model_pref_set_last(const char *file)
{
    if (!file || !*file) {
        return -1;
    }
    const char *base = strrchr(file, '/');
    base = base ? base + 1 : file;
    size_t len = strlen(base);
    if (len == 0 || len >= MODEL_PREF_NAME_MAX ||
        strncmp(base, "partition:", 10) == 0) {
        return -1;
    }

    nvs_handle_t h;
    if (!prefs_open(&h, NVS_READWRITE)) {
        return -1;
    }
    esp_err_t err = nvs_set_str(h, PREF_KEY, base);
    if (err == ESP_OK) {
        err = nvs_commit(h);
    }
    prefs_close(h);
    if (err != ESP_OK) {
        ESP_LOGW(TAG, "nvs_set_str: %s", esp_err_to_name(err));
        return -1;
    }
    ESP_LOGI(TAG, "last model = %s", base);
    return 0;
}

void model_pref_clear(void)
{
    nvs_handle_t h;
    if (!prefs_open(&h, NVS_READWRITE)) {
        return;
    }
    esp_err_t err = nvs_erase_key(h, PREF_KEY);
    if (err == ESP_OK) {
        err = nvs_commit(h);
    }
    prefs_close(h);
    if (err != ESP_OK && err != ESP_ERR_NVS_NOT_FOUND) {
        ESP_LOGW(TAG, "nvs_erase_key: %s", esp_err_to_name(err));
    }
}
