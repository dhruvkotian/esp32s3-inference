// ESP32-S3 int8 dot product using the 128-bit vector registers.
//
// The S3 vector unit can multiply-accumulate 16 signed 8-bit lanes per
// instruction (EE.VMULAS.S8.ACCX) and keeps the running sum in the 40-bit
// ACCX accumulator. That is 16 MACs per instruction versus 1 for the scalar
// path, which is the whole point of this experiment.
//
// int32_t llm_dot_s8(const int8_t *a, const int8_t *b, int n)
//   a2 = a, a3 = b, a4 = n
//   Both pointers must be 16-byte aligned and n should be a multiple of 16
//   for the vector path. A scalar tail handles any remainder.

    .text
    .align 4
    .global llm_dot_s8
    .type   llm_dot_s8,@function

llm_dot_s8:
    entry   a1, 32
    movi    a5, 0               // accumulator for the scalar tail
    srli    a6, a4, 4           // number of 16-byte blocks
    beqz    a6, .Ltail

    movi    a14, 0
    wur.accx_0 a14              // clear ACCX
    wur.accx_1 a14

    // Fused load and MAC. EE.VMULAS.S8.ACCX.LD.IP accumulates the current
    // 16-byte block and loads the next b block in the same instruction, so the
    // inner loop is two instructions and uses the zero-overhead loop.
    ee.vld.128.ip q0, a2, 16
    ee.vld.128.ip q1, a3, 16
    addi    a6, a6, -1
    beqz    a6, .Llast
    loopnez a6, .Lblk
    ee.vmulas.s8.accx.ld.ip q1, a3, 16, q0, q1
    ee.vld.128.ip q0, a2, 16
.Lblk:
.Llast:
    ee.vmulas.s8.accx q0, q1    // ACCX += sum(a[i] * b[i]) for 16 lanes

    rur.accx_0 a5               // low 32 bits of ACCX

.Ltail:
    extui   a7, a4, 0, 4        // n & 15
    beqz    a7, .Ldone
.Lt:
    l8ui    a8, a2, 0
    l8ui    a9, a3, 0
    sext    a8, a8, 7
    sext    a9, a9, 7
    mull    a8, a8, a9
    add     a5, a5, a8
    addi    a2, a2, 1
    addi    a3, a3, 1
    addi    a7, a7, -1
    bnez    a7, .Lt

.Ldone:
    mov     a2, a5
    retw.n
    .size   llm_dot_s8, . - llm_dot_s8


// Single 16-lane int8 dot, no loop and no scalar tail.
//
// int32_t llm_dot16_s8(const int8_t *a, const int8_t *b)
//   a2 = a, a3 = b. Both must be 16-byte aligned. Returns the int32 sum of
//   the 16 signed products. Used for the attention query-key dot, where one
//   head is 8 values and the other 8 lanes are zero padding.
    .align 4
    .global llm_dot16_s8
    .type   llm_dot16_s8,@function

llm_dot16_s8:
    entry   a1, 32
    movi    a14, 0
    wur.accx_0 a14
    wur.accx_1 a14
    ee.vld.128.ip q0, a2, 16
    ee.vld.128.ip q1, a3, 16
    ee.vmulas.s8.accx q0, q1
    rur.accx_0 a2
    retw.n
    .size   llm_dot16_s8, . - llm_dot16_s8


// Four 16-lane int8 dots against consecutive K records.
//
// void llm_dot16x4_s8(const int8_t *a, const int8_t *b, int stride,
//                     int32_t out[4])
//   a2 = a (query, 16 B), a3 = b (first K chunk), a4 = stride in bytes,
//   a5 = out. Each dot uses the same query, so q0 is loaded once. The K
//   chunks are `stride` bytes apart, which is kv_dim in the caller.
    .align 4
    .global llm_dot16x4_s8
    .type   llm_dot16x4_s8,@function

llm_dot16x4_s8:
    entry   a1, 32
    ee.vld.128.ip q0, a2, 0

    movi    a14, 0
    wur.accx_0 a14
    wur.accx_1 a14
    ee.vld.128.ip q1, a3, 0
    ee.vmulas.s8.accx q0, q1
    rur.accx_0 a6
    s32i    a6, a5, 0
    add     a3, a3, a4

    movi    a14, 0
    wur.accx_0 a14
    wur.accx_1 a14
    ee.vld.128.ip q1, a3, 0
    ee.vmulas.s8.accx q0, q1
    rur.accx_0 a6
    s32i    a6, a5, 4
    add     a3, a3, a4

    movi    a14, 0
    wur.accx_0 a14
    wur.accx_1 a14
    ee.vld.128.ip q1, a3, 0
    ee.vmulas.s8.accx q0, q1
    rur.accx_0 a6
    s32i    a6, a5, 8
    add     a3, a3, a4

    movi    a14, 0
    wur.accx_0 a14
    wur.accx_1 a14
    ee.vld.128.ip q1, a3, 0
    ee.vmulas.s8.accx q0, q1
    rur.accx_0 a6
    s32i    a6, a5, 12

    retw.n
    .size   llm_dot16x4_s8, . - llm_dot16x4_s8


// Scalar reference used to self-test the vector kernel at boot.
// int32_t llm_dot_s8_scalar(const int8_t *a, const int8_t *b, int n)
    .align 4
    .global llm_dot_s8_scalar
    .type   llm_dot_s8_scalar,@function

llm_dot_s8_scalar:
    entry   a1, 32
    movi    a5, 0
    beqz    a4, .Lsdone
.Ls:
    l8ui    a8, a2, 0
    l8ui    a9, a3, 0
    sext    a8, a8, 7
    sext    a9, a9, 7
    mull    a8, a8, a9
    add     a5, a5, a8
    addi    a2, a2, 1
    addi    a3, a3, 1
    addi    a4, a4, -1
    bnez    a4, .Ls
.Lsdone:
    mov     a2, a5
    retw.n
    .size   llm_dot_s8_scalar, . - llm_dot_s8_scalar


// Fused load+MAC probe.
//
// EE.VMULAS.S8.ACCX.LD.IP takes (qu, as, imm16, qx, qy): it loads 16 bytes
// from as into qu, advances as by imm16, and does ACCX += qx * qy. The load
// target is independent of the multiply sources, so a fixed activation can
// live in q0..q3 while the weights stream in through q4..q7 with one
// instruction per 16 bytes instead of two.
//
// int32_t llm_dot_s8_fused64(const int8_t *a, const int8_t *b)
//   a2 = a (activation, 64 bytes), a3 = b (weights, 64 bytes)
//   Both must be 16-byte aligned. Used to prove the encoding works before
//   the matmul loop depends on it.
    .align 4
    .global llm_dot_s8_fused64
    .type   llm_dot_s8_fused64,@function

llm_dot_s8_fused64:
    entry   a1, 32
    movi    a14, 0
    wur.accx_0 a14
    wur.accx_1 a14

    ee.vld.128.ip q0, a2, 16
    ee.vld.128.ip q1, a2, 16
    ee.vld.128.ip q2, a2, 16
    ee.vld.128.ip q3, a2, 16

    ee.vld.128.ip q4, a3, 16
    ee.vmulas.s8.accx.ld.ip q5, a3, 16, q0, q4
    ee.vmulas.s8.accx.ld.ip q6, a3, 16, q1, q5
    ee.vmulas.s8.accx.ld.ip q7, a3, 16, q2, q6
    ee.vmulas.s8.accx q3, q7

    rur.accx_0 a2
    retw.n
    .size   llm_dot_s8_fused64, . - llm_dot_s8_fused64


// Stride-64 matvec with the activation resident in vector registers.
//
// void llm_mm_s8_64(const int8_t *act, const int8_t *w, const float *scale,
//                   const float *xs, float *out, int nrows)
//
// Every stride-64 weight row is exactly four 16-byte blocks, so the whole
// 64-byte activation fits in q0..q3 and stays there for the entire row loop.
// Each row then costs one plain load plus three fused load+MACs plus one
// final MAC, instead of four loads and four MACs.
    .align 4
    .global llm_mm_s8_64
    .type   llm_mm_s8_64,@function

llm_mm_s8_64:
    entry   a1, 32

    ee.vld.128.ip q0, a2, 16
    ee.vld.128.ip q1, a2, 16
    ee.vld.128.ip q2, a2, 16
    ee.vld.128.ip q3, a2, 16

    lsi     f2, a5, 0               // xs
    blti    a7, 1, .Lmm64_done

.Lmm64_row:
    movi    a14, 0
    wur.accx_0 a14
    wur.accx_1 a14

    ee.vld.128.ip q4, a3, 16
    ee.vmulas.s8.accx.ld.ip q5, a3, 16, q0, q4
    ee.vmulas.s8.accx.ld.ip q6, a3, 16, q1, q5
    ee.vmulas.s8.accx.ld.ip q7, a3, 16, q2, q6
    ee.vmulas.s8.accx q3, q7

    rur.accx_0 a8
    float.s f0, a8, 0
    lsi     f1, a4, 0               // row scale
    mul.s   f0, f0, f2
    mul.s   f0, f0, f1
    ssi     f0, a6, 0

    addi    a4, a4, 4
    addi    a6, a6, 4
    addi    a7, a7, -1
    bnez    a7, .Lmm64_row

.Lmm64_done:
    retw.n
    .size   llm_mm_s8_64, . - llm_mm_s8_64

