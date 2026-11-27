#pragma once

// Standalone PSRAM-resident expert access benchmark (v1.4.0 investigation).
//
// Builds its own synthetic expert blocks with the d=96 / hidden=264 geometry.
// It needs no model, no MOE1 sidecar and no SD card, and it does not touch
// moe_hook(). Enable with EXPERT_ACCESS_BENCH in bench_cfg.h.
void expert_access_bench_run(void);
