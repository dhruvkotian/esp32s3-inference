#pragma once

#include <stddef.h>

// Last successfully selected model file, persisted in NVS.
//
// Only the basename is stored (for example "alt.m8"), never a full path and
// never a "partition:*" pseudo-path: the flash partition is a fallback, not a
// user selection, and the card mount point is fixed.
//
// NVS is opened and closed on every call so its page cache does not stay in
// internal SRAM while the int8 arena is live. That matters on this board: a
// few kilobytes of internal SRAM decide whether a weight matrix stays in SRAM
// or spills to PSRAM.

#define MODEL_PREF_NAME_MAX 64

// 0 when a stored selection was copied to `out`; -1 when NVS is empty or
// unreadable. `out` is an empty string on failure.
int model_pref_get_last(char *out, size_t cap);

// Store the basename of `file`. Returns 0 on success, -1 on failure or when
// the name is empty / a partition pseudo-path.
int model_pref_set_last(const char *file);

// Forget the stored selection. Used by tests and factory reset paths.
void model_pref_clear(void);
