#pragma once
#define BENCH_MODE 1
#define HOT_POS 16
#define WEIGHTS_PSRAM 0
#define DUMP_LOGITS 1
// Run the nine-depth accuracy harness. Independent of KV_INT8 so the int8
// weight path can be measured against fp32 KV as well as int8 KV.
#define KVCHECK 1
#define VAR_RUNS 10
#define VAR_TOKENS 12
#define CTX_RUNS 3
#define CTX_LEN 512

// Experiment: store K/V in PSRAM as int8 with one fp32 scale per head per
// position. The hot window stays fp32. Build with 0 for the baseline and 1
// for the INT8 KV variant; everything else is unchanged.
#define KV_INT8 1

// Experiment: split the attention heads across both cores. The worker core
// handles the upper half of the heads with its own softmax scratch, one
// notification pair per layer.
#define DUAL_CORE 1

// Experiment: prefetch the int8 V cache into internal SRAM with AHB-GDMA.
// V is stored in position tiles so one DMA transaction covers a whole tile for
// all 8 dimensions. Build with DUAL_CORE 0 to isolate the DMA effect.
#define DMA_PREFETCH 0
#define DMA_ACTIVE 1
#define V_TILE 64
