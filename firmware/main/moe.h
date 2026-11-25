#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "llm.h"

// Synthetic MoE prototype: FFN-only experts layered onto a loaded int8 model.
//
// A sidecar file supplies N alternative FFN parameter sets. The backbone
// (embedding, attention, norms) stays the one in the active .m8 model. At each
// layer the router picks an expert, and the matching w1/w2/w3 row range is
// copied from a PSRAM-resident sidecar into the existing internal-SRAM int8
// arena. Arena shapes never change.
//
// This is a mechanism and cost experiment, not a trained MoE. Experts are
// extracted from packed .m8 models and the router is deterministic synthetic
// data, so quality and routing claims are explicitly out of scope.
//
// Wire format (little-endian, 4 KiB header, mirrors the .m8 conventions):
//
//   0    magic "MOE1"          136  name[32]
//   4    u16 version           168  build[32]
//   6    u16 header_bytes
//   8    u32 file_bytes
//   12   u32 payload_bytes
//   16   u32 flags (bit0 router is f32, bit1 synthetic)
//   20   u32 header_crc32 (over the header with this field zero)
//   24   u8  payload_sha256[32]
//   56   u16 dim
//   58   u16 hidden
//   60   u16 layers
//   62   u16 experts
//   64   u8  row_align
//   65   u8[3] reserved
//   68   u32 model_sanity_crc  (sanity fingerprint of expert 0's source model)
//   72   u32 router_off        (payload-relative)
//   76   u32 router_bytes
//   80   u32 expert_bytes      (size of one expert block)
//   84   u32 expert_off        (payload-relative, expert e at off + e*bytes)
//   88   u32 w1_q_off;  92 w1_q_bytes
//   96   u32 w1_s_off; 100 w1_s_bytes
//   104  u32 w2_q_off; 108 w2_q_bytes
//   112  u32 w2_s_off; 116 w2_s_bytes
//   120  u32 w3_q_off; 124 w3_q_bytes
//   128  u32 w3_s_off; 132 w3_s_bytes
//
// Router weights are float32 [layer][expert][dim]. Expert blocks contain whole
// tensors in the same row-major, 16-byte-strided int8 form as the .m8 arena, so
// a per-layer update is a memcpy of each tensor's row range.

#define MOE_MAGIC        "MOE1"
#define MOE_VERSION      1
#define MOE_HEADER_BYTES 4096
#define MOE_MAX_EXPERTS  4
#define MOE_MAX_LAYERS   16
#define MOE_MAX_BYTES    (4 * 1024 * 1024)

typedef enum {
    MOE_OK = 0,
    MOE_ERR_IO = -1,
    MOE_ERR_SIZE = -2,
    MOE_ERR_MAGIC = -3,
    MOE_ERR_VERSION = -4,
    MOE_ERR_HEADER_CRC = -5,
    MOE_ERR_SHA = -6,
    MOE_ERR_SHAPE = -7,
    MOE_ERR_NOMEM = -8,
    MOE_ERR_UNBOUND = -9,
} moe_err_t;

typedef struct {
    uint8_t *blob;          // whole file, PSRAM, owned
    size_t file_bytes;
    char path[64];
    char name[32];
    char build[32];
    uint16_t dim, hidden, layers, experts;
    uint32_t model_sanity_crc;
    uint32_t router_off, router_bytes;
    uint32_t expert_off, expert_bytes;
    uint32_t w1q_off, w1q_bytes, w1s_off, w1s_bytes;
    uint32_t w2q_off, w2q_bytes, w2s_off, w2s_bytes;
    uint32_t w3q_off, w3q_bytes, w3s_off, w3s_bytes;
} moe_stage_t;

typedef struct {
    uint32_t route_calls;       // hook invocations (layers x forwards)
    uint32_t copies;            // per-layer arena refills
    uint32_t copy_bytes;        // bytes moved by those refills
    uint64_t copy_us;           // wall time inside the refills
    uint32_t route_count[MOE_MAX_EXPERTS];
    int last_expert[MOE_MAX_LAYERS];
    int experts;
    int layers;
} moe_stats_t;

// Load and validate a sidecar from FATFS into PSRAM.
int moe_stage_file(const char *path, moe_stage_t *st);

// Load a length-prefixed sidecar image from a flash data partition.
int moe_stage_partition(const char *label, moe_stage_t *st);

// Write a staged sidecar to FATFS. Used by the test provisioning path.
int moe_stage_save(const moe_stage_t *st, const char *path);

void moe_stage_free(moe_stage_t *st);

// Attach a validated sidecar to a live int8 model and install the per-layer
// hook. Consumes *st on success and on failure. Returns MOE_ERR_SHAPE if the
// sidecar geometry or arena layout does not match the model.
int moe_bind(llm_t *m, moe_stage_t *st);

// Remove the hook and free the sidecar and its state.
void moe_unbind(llm_t *m);

// expert = -1 uses the router dot product; 0..E-1 pins one expert and disables
// round-robin. moe_set_round_robin() alternates per layer on every call, which
// is the worst case for copy traffic.
void moe_force_expert(llm_t *m, int expert);
void moe_set_round_robin(llm_t *m, bool on);

void moe_get_stats(const llm_t *m, moe_stats_t *out);
void moe_reset_stats(llm_t *m);
const char *moe_strerror(int rc);
