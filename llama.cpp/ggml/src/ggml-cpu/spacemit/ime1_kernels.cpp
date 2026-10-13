#include "ggml-impl.h"
#include "ggml.h"
#include "ime_kernels.h"
#include "rvv_kernels.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#if !defined(__riscv_v) || !defined(__riscv_v_intrinsic)
#    error "riscv v extension or v_intrinsic not enabled"
#else
#    include <riscv_vector.h>
#endif

#if !defined(__riscv_zfh)
#    error "riscv zfh extension not enabled"
#endif

#if defined(RISCV64_SPACEMIT_IME1)
#else
#    error "RISCV64_SPACEMIT_IME1 not defined"
#endif

// clang-format off
#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Woverlength-strings"
#pragma GCC diagnostic ignored "-Wcast-qual"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
// clang-format on
namespace spacemit_kernels {

#define QUANTIZEM4ROW_KERNEL                           \
    "vmv.s.x            v16, zero                \n\t" \
    "vfabs.v            v8, v0                   \n\t" \
    "vfredmax.vs        v16, v8, v16             \n\t" \
    "vfmv.f.s           f10, v16                 \n\t" \
    "fmul.s             f10, f10, %[RMAXREC]     \n\t" \
    "fsw                f10, (a1)                \n\t" \
    "fdiv.s             f11, %[FONE], f10        \n\t" \
    "vfmul.vf           v16, v0, f11             \n\t" \
    "vfcvt.x.f.v        v16, v16                 \n\t" \
    "vsetvli            t0, zero, e16, mf2       \n\t" \
    "vnclip.wx          v16, v16, zero           \n\t" \
    "vnclip.wx          v17, v17, zero           \n\t" \
    "vnclip.wx          v18, v18, zero           \n\t" \
    "vnclip.wx          v19, v19, zero           \n\t" \
    "vnclip.wx          v20, v20, zero           \n\t" \
    "vnclip.wx          v21, v21, zero           \n\t" \
    "vnclip.wx          v22, v22, zero           \n\t" \
    "vnclip.wx          v23, v23, zero           \n\t" \
    "vsetvli            t0, zero, e8, mf4        \n\t" \
    "vnclip.wx          v24, v16, zero           \n\t" \
    "vnclip.wx          v25, v17, zero           \n\t" \
    "vnclip.wx          v26, v18, zero           \n\t" \
    "vnclip.wx          v27, v19, zero           \n\t" \
    "vnclip.wx          v28, v20, zero           \n\t" \
    "vnclip.wx          v29, v21, zero           \n\t" \
    "vnclip.wx          v30, v22, zero           \n\t" \
    "vnclip.wx          v31, v23, zero           \n\t"

#define QUANTIZEM4ROW_STORE                            \
    "addi               t1, %[BlkLen], 0         \n\t" \
    "vsetvli            t0, t1, e8, mf4          \n\t" \
    "vse8.v             v24, (s1)                \n\t" \
    "addi               s1, s1, 32               \n\t" \
    "sub                t1, t1, t0               \n\t" \
    "vsetvli            t0, t1, e8, mf4          \n\t" \
    "vse8.v             v25, (s1)                \n\t" \
    "addi               s1, s1, 32               \n\t" \
    "sub                t1, t1, t0               \n\t" \
    "vsetvli            t0, t1, e8, mf4          \n\t" \
    "vse8.v             v26, (s1)                \n\t" \
    "addi               s1, s1, 32               \n\t" \
    "sub                t1, t1, t0               \n\t" \
    "vsetvli            t0, t1, e8, mf4          \n\t" \
    "vse8.v             v27, (s1)                \n\t" \
    "addi               s1, s1, 32               \n\t" \
    "sub                t1, t1, t0               \n\t" \
    "vsetvli            t0, t1, e8, mf4          \n\t" \
    "vse8.v             v28, (s1)                \n\t" \
    "addi               s1, s1, 32               \n\t" \
    "sub                t1, t1, t0               \n\t" \
    "vsetvli            t0, t1, e8, mf4          \n\t" \
    "vse8.v             v29, (s1)                \n\t" \
    "addi               s1, s1, 32               \n\t" \
    "sub                t1, t1, t0               \n\t" \
    "vsetvli            t0, t1, e8, mf4          \n\t" \
    "vse8.v             v30, (s1)                \n\t" \
    "addi               s1, s1, 32               \n\t" \
    "sub                t1, t1, t0               \n\t" \
    "vsetvli            t0, t1, e8, mf4          \n\t" \
    "vse8.v             v31, (s1)                \n\t"

namespace ime1 {
// [Q4_0_K128] BlkLen=128 activation quantiser for the IME1 M4 path.
//
// The vectorised BlkLen=64 routine above cannot form a 128-wide amax (e32/m8 caps VL at
// 64), so this variant takes the max over two 64-lane passes and then quantises in
// 8-value groups, which matches the payload layout the kernel reads:
//   per 128-value block and 4-row group: [4 x f32 scales][512 B payload];
//   payload group g (8 values per row) sits at 16 + 32g, row r at +8r.
// Arithmetic mirrors the assembly exactly: scale = amax * (1/127); q = rne(a/scale)
// with a real division by the scale; saturating narrowing to int8.
static void quantize_a_4row_i8_k128(const float * GGML_RESTRICT A, size_t CountK, uint8_t * GGML_RESTRICT QuantA) {
    constexpr size_t  BlkLen                = 128;
    constexpr float   range_max_reciprocal  = 1.0f / ((1 << 7) - 1);

    GGML_ASSERT(CountK % BlkLen == 0);
    const size_t nblk = CountK / BlkLen;

    for (size_t b = 0; b < nblk; ++b) {
        uint8_t * dst = QuantA + b * 4 * (sizeof(float) + BlkLen);

        for (int r = 0; r < 4; ++r) {
            const float * src = A + r * CountK + b * BlkLen;

            const vfloat32m8_t v0 = __riscv_vle32_v_f32m8(src,      64);
            const vfloat32m8_t v1 = __riscv_vle32_v_f32m8(src + 64, 64);
            const vfloat32m8_t vm = __riscv_vfmax_vv_f32m8(__riscv_vfabs_v_f32m8(v0, 64),
                                                           __riscv_vfabs_v_f32m8(v1, 64), 64);
            const vfloat32m1_t red =
                __riscv_vfredmax_vs_f32m8_f32m1(vm, __riscv_vfmv_s_f_f32m1(0.0f, 1), 64);
            const float amax = __riscv_vfmv_f_s_f32m1_f32(red);

            const float scale = amax * range_max_reciprocal;
            const float inv   = scale != 0.0f ? 1.0f / scale : 0.0f;
            memcpy(dst + r * sizeof(float), &scale, sizeof(float));

            uint8_t * out = dst + 16 + 8 * r;
            for (size_t g = 0; g < BlkLen / 8; ++g) {
                const vfloat32m1_t v = __riscv_vle32_v_f32m1(src + g * 8, 8);
                const vfloat32m1_t q = __riscv_vfmul_vf_f32m1(v, inv, 8);
                const vint32m1_t   i = __riscv_vfcvt_x_f_v_i32m1(q, 8);
                const vint16mf2_t  h = __riscv_vnclip_wx_i16mf2(i, 0, 0, 8);
                const vint8mf4_t   c = __riscv_vnclip_wx_i8mf4(h, 0, 0, 8);
                __riscv_vse8_v_i8mf4((int8_t *) (out + 32 * g), c, 8);
            }
        }
    }
}

// [A-offset fold] quantize_a_4row_i8_k128 plus the sum of the quantised int8 values over each
// 128-value K block, for each of the 4 rows.  The sums are stored as the two int32
// row-broadcast patterns the folded kernel subtracts, so the record becomes 592 B:
//     [16 B scales][512 B payload][32 B Sa][32 B Sb]
//     Sa = [8*S0 x4 | 8*S1 x4]   Sb = [8*S2 x4 | 8*S3 x4]
// (row order follows SAVE_RESULT_4x16: even accumulator registers carry rows 0/1, odd ones
//  rows 2/3).  A block stride therefore becomes 148 B per row per K block.
//
// Why this is bit-identical: the kernel computes sum_k a*(q-8) = sum_k a*q - 8*sum_k a.
// `a` here is the stored int8 value.  The scale is amax/127, so |i| <= 127 is guaranteed and
// the int32 -> int16 -> int8 narrowing below never saturates: summing `i` is exactly summing
// the stored int8 values.  All of it is integer arithmetic.
static void quantize_a_4row_i8_k128_sum(const float * GGML_RESTRICT A, size_t CountK, uint8_t * GGML_RESTRICT QuantA) {
    constexpr size_t BlkLen    = 128;
    // [16 B row scales][4 rows x 128 B payload][Sa (8 x int32)][Sb (8 x int32)] = 592 B,
    // i.e. 148 B per row per K block -- must match spx_a_fold_extra_per_blk() in ime.cpp
    constexpr size_t BlkStride = 4 * (sizeof(float) + BlkLen) + 2 * 8 * sizeof(int32_t);
    constexpr float  range_max_reciprocal = 1.0f / ((1 << 7) - 1);

    GGML_ASSERT(CountK % BlkLen == 0);
    const size_t nblk = CountK / BlkLen;

    for (size_t b = 0; b < nblk; ++b) {
        uint8_t * dst = QuantA + b * BlkStride;
        int32_t   srow[4] = { 0, 0, 0, 0 };

        for (int r = 0; r < 4; ++r) {
            const float * src = A + r * CountK + b * BlkLen;

            const vfloat32m8_t v0 = __riscv_vle32_v_f32m8(src,      64);
            const vfloat32m8_t v1 = __riscv_vle32_v_f32m8(src + 64, 64);
            const vfloat32m8_t vm = __riscv_vfmax_vv_f32m8(__riscv_vfabs_v_f32m8(v0, 64),
                                                           __riscv_vfabs_v_f32m8(v1, 64), 64);
            const vfloat32m1_t red =
                __riscv_vfredmax_vs_f32m8_f32m1(vm, __riscv_vfmv_s_f_f32m1(0.0f, 1), 64);
            const float amax = __riscv_vfmv_f_s_f32m1_f32(red);

            const float scale = amax * range_max_reciprocal;
            const float inv   = scale != 0.0f ? 1.0f / scale : 0.0f;
            memcpy(dst + r * sizeof(float), &scale, sizeof(float));

            uint8_t * out = dst + 16 + 8 * r;
            vint32m1_t sum = __riscv_vmv_v_x_i32m1(0, 8);
            for (size_t g = 0; g < BlkLen / 8; ++g) {
                const vfloat32m1_t v = __riscv_vle32_v_f32m1(src + g * 8, 8);
                const vfloat32m1_t q = __riscv_vfmul_vf_f32m1(v, inv, 8);
                const vint32m1_t   i = __riscv_vfcvt_x_f_v_i32m1(q, 8);
                const vint16mf2_t  h = __riscv_vnclip_wx_i16mf2(i, 0, 0, 8);
                const vint8mf4_t   c = __riscv_vnclip_wx_i8mf4(h, 0, 0, 8);
                sum = __riscv_vadd_vv_i32m1(sum, i, 8);
                __riscv_vse8_v_i8mf4((int8_t *) (out + 32 * g), c, 8);
            }
            const vint32m1_t rsum = __riscv_vredsum_vs_i32m1_i32m1(sum, __riscv_vmv_v_x_i32m1(0, 1), 8);
            srow[r] = __riscv_vmv_x_s_i32m1_i32(rsum) * 8;
        }
        int32_t * pat = (int32_t *) (dst + 16 + 4 * BlkLen);  // right after the 4 rows x 128 B payload
        __riscv_vse32_v_i32m1(pat + 0, __riscv_vmv_v_x_i32m1(srow[0], 4), 4);
        __riscv_vse32_v_i32m1(pat + 4, __riscv_vmv_v_x_i32m1(srow[1], 4), 4);
        __riscv_vse32_v_i32m1(pat + 8, __riscv_vmv_v_x_i32m1(srow[2], 4), 4);
        __riscv_vse32_v_i32m1(pat + 12, __riscv_vmv_v_x_i32m1(srow[3], 4), 4);
    }
}

void quantize_a_4row_i8_with_sum(size_t BlkLen, const float * A, size_t CountK, uint8_t * QuantA) {
    GGML_ASSERT(BlkLen == 128);
    quantize_a_4row_i8_k128_sum(A, CountK, QuantA);
}

void quantize_a_4row_i8(size_t BlkLen, const float * A, size_t CountK, uint8_t * QuantA) {
    if (BlkLen == 128) { // QK4_0_K128
        quantize_a_4row_i8_k128(A, CountK, QuantA);
        return;
    }
    constexpr float range_max_reciprocal = 1.0f / ((1 << 7) - 1);
    const float     fone                 = 1.0f;

    for (size_t row_index = 0; row_index < 4; ++row_index) {
        const float * SRC = A + row_index * CountK;
        uint8_t *     DST = QuantA + row_index * sizeof(float);

        const size_t offset = (4 - row_index) * 4 + row_index * 8;
        const size_t stride = 4 * (sizeof(float) + BlkLen);
        __asm__ volatile(
            "vsetvli            t0, zero, e32, m8        \n\t"
            "addi               t2, %[CountK], 0         \n\t"
            "addi               a1, %[DST], 0            \n\t"
            "blt                t2, %[BlkLen], TAIL%=    \n\t"

            "LOOP%=:                                     \n\t"
            "vsetvli            t0, %[BlkLen], e32, m8   \n\t"
            "vle32.v            v0, (%[SRC])             \n\t"
            "sub                t2, t2, t0               \n\t"
            "slli               t1, t0, 2                \n\t"
            "add                %[SRC], %[SRC], t1       \n\t"
            "add                s1, a1, %[OFFSET]        \n\t"

            QUANTIZEM4ROW_KERNEL QUANTIZEM4ROW_STORE

            "add                a1, a1, %[STRIDE]        \n\t"
            "bge                t2, %[BlkLen], LOOP%=    \n\t"

            "TAIL%=:                                     \n\t"
            "blez               t2, QUIT%=               \n\t"
            "vsetvli            t0, zero, e32, m8        \n\t"
            "vxor.vv            v16, v16, v16            \n\t"
            "vxor.vv            v24, v24, v24            \n\t"
            "vsetvli            t0, t2, e32, m8          \n\t"
            "vle32.v            v0, (%[SRC])             \n\t"
            "add                s1, a1, %[OFFSET]        \n\t"

            QUANTIZEM4ROW_KERNEL

            "addi               t3, %[BlkLen], 0         \n\t"
            "addi               s2, s1, 0                \n\t"
            "vsetvli            t0, zero, e8, mf4        \n\t"
            "vxor.vv            v8, v8, v8               \n\t"
            "SET_ZERO%=:                                 \n\t"
            "vse8.v             v8, (s2)                 \n\t"
            "addi               s2, s2, 32               \n\t"
            "addi               t3, t3, -8               \n\t"
            "bnez               t3, SET_ZERO%=           \n\t"

            QUANTIZEM4ROW_STORE

            "QUIT%=:                                     \n\t"
            : [SRC] "+r"(SRC)
            : [DST] "r"(DST), [BlkLen] "r"(BlkLen), [OFFSET] "r"(offset), [STRIDE] "r"(stride), [CountK] "r"(CountK),
              [FONE] "f"(fone), [RMAXREC] "f"(range_max_reciprocal)
            : "cc", "t0", "t1", "t2", "t3", "a1", "s1", "s2", "f10", "f11");
    }
}

// [Q4_0_K64] BlkLen-generic single-row A quantiser for the IME1 path.
//
// The vector routine below hard-codes BlkLen = 32 in its assembly (block
// stride 36, 128-element tiles), so it produces a wrong A layout when called
// with BlkLen = 64 -- which is what the gemm_m == 1 (decode) branch does.
// This scalar version reproduces the assembly's arithmetic exactly:
//   scale = amax * (1/127);  q = rne(a / scale);  saturate to int8
// and writes the same per-block layout: [f32 scale][BlkLen int8], stride
// 4 + BlkLen. Only reached for BlkLen != 32; the cost is O(K) against an
// O(K*N) GEMM.
static void quantize_a_row_i8_ref(size_t BlkLen, const float * A, size_t CountK, uint8_t * QuantA) {
    const float  range_max_reciprocal = 1.0f / ((1 << 7) - 1);
    const float  fone                 = 1.0f;
    const size_t nblk                 = (CountK + BlkLen - 1) / BlkLen;

    for (size_t b = 0; b < nblk; ++b) {
        const size_t off = b * BlkLen;
        const size_t n   = (off + BlkLen <= CountK) ? BlkLen : (CountK - off);

        float amax = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            const float v = fabsf(A[off + i]);
            if (v > amax) {
                amax = v;
            }
        }

        const float scale = amax * range_max_reciprocal;
        const float rep   = scale ? fone / scale : 0.0f;

        uint8_t * dst = QuantA + b * (sizeof(float) + BlkLen);
        memcpy(dst, &scale, sizeof(float));

        int8_t * q = reinterpret_cast<int8_t *>(dst + sizeof(float));
        for (size_t i = 0; i < n; ++i) {
            long v = lrintf(A[off + i] * rep);
            if (v >  127) { v =  127; }
            if (v < -128) { v = -128; }
            q[i] = static_cast<int8_t>(v);
        }
        for (size_t i = n; i < BlkLen; ++i) {
            q[i] = 0;
        }
    }
}

void quantize_a_row_i8(size_t BlkLen, const float * A, size_t CountK, uint8_t * QuantA) {
    if (BlkLen != 32) {
        quantize_a_row_i8_ref(BlkLen, A, CountK, QuantA);
        return;
    }
    const float *   SRC                  = A;
    uint8_t *       DST                  = QuantA;
    constexpr float range_max_reciprocal = 1.0f / ((1 << 7) - 1);
    const float     fone                 = 1.0f;
    uint8_t *       QuantA_offset        = QuantA + CountK + 4 * ((CountK + BlkLen - 1) / BlkLen);
    size_t          offset               = (CountK + BlkLen - 1) / BlkLen * BlkLen - CountK;

    __asm__ volatile(
        "addi         t3, zero, 32*4          \n\t"
        "addi         t2, zero, 32            \n\t"

        "addi         a1, %[SRC], 0           \n\t"
        "addi         a2, %[SRC], 128         \n\t"
        "addi         a3, %[SRC], 256         \n\t"
        "addi         a4, %[SRC], 384         \n\t"

        "addi         s1, %[DST], 0           \n\t"
        "addi         s2, %[DST], 36          \n\t"
        "addi         s3, %[DST], 72          \n\t"
        "addi         s4, %[DST], 108         \n\t"
        "blt          %[K], t3, LOOP_K%=      \n\t"
        "blt          %[K], t2, TAIL%=        \n\t"

        "LOOP_MAIN%=:                         \n\t"
        "vsetvli      t1, zero, e32, m4       \n\t"
        "addi         %[K], %[K], -128        \n\t"
        "vle32.v      v0, (a1)                \n\t"
        "addi         a1, a1, 512             \n\t"
        "vle32.v      v4, (a2)                \n\t"
        "addi         a2, a2, 512             \n\t"
        "vle32.v      v8, (a3)                \n\t"
        "addi         a3, a3, 512             \n\t"
        "vle32.v      v12, (a4)               \n\t"
        "addi         a4, a4, 512             \n\t"
        "vfabs.v      v16, v0                 \n\t"
        "vfabs.v      v20, v4                 \n\t"
        "vfabs.v      v24, v8                 \n\t"
        "vfabs.v      v28, v12                \n\t"
        "vsetvli      t0, zero, e32, m2       \n\t"
        "vfmax.vv     v16, v16, v18           \n\t"
        "vfmax.vv     v20, v20, v22           \n\t"
        "vfmax.vv     v24, v24, v26           \n\t"
        "vfmax.vv     v28, v28, v30           \n\t"
        "vsetvli      t0, zero, e32, m1       \n\t"
        "vfmax.vv     v16, v16, v17           \n\t"
        "vfmax.vv     v20, v20, v21           \n\t"
        "vfmax.vv     v24, v24, v25           \n\t"
        "vfmax.vv     v28, v28, v29           \n\t"

        "vfredmax.vs  v17, v16, v17           \n\t"
        "vfredmax.vs  v21, v20, v21           \n\t"
        "vfredmax.vs  v25, v24, v25           \n\t"
        "vfredmax.vs  v29, v28, v29           \n\t"
        "vfmv.f.s     f10,  v17               \n\t"
        "vfmv.f.s     f11,  v21               \n\t"
        "vfmv.f.s     f12,  v25               \n\t"
        "vfmv.f.s     f13,  v29               \n\t"

        "fmul.s       f10, f10, %[RMAXREC]    \n\t"
        "fmul.s       f11, f11, %[RMAXREC]    \n\t"
        "fmul.s       f12, f12, %[RMAXREC]    \n\t"
        "fmul.s       f13, f13, %[RMAXREC]    \n\t"
        "fsw          f10, (s1)               \n\t"
        "addi         s1, s1, 4               \n\t"

        "fsw          f11, (s2)               \n\t"
        "addi         s2, s2, 4               \n\t"
        "fsw          f12, (s3)               \n\t"
        "addi         s3, s3, 4               \n\t"
        "fsw          f13, (s4)               \n\t"
        "addi         s4, s4, 4               \n\t"
        "fdiv.s       f10, %[FONE], f10       \n\t"
        "fdiv.s       f11, %[FONE], f11       \n\t"
        "fdiv.s       f12, %[FONE], f12       \n\t"
        "fdiv.s       f13, %[FONE], f13       \n\t"
        "vsetvli      t0, zero, e32, m4       \n\t"
        "vfmul.vf     v16, v0, f10            \n\t"
        "vfmul.vf     v20, v4, f11            \n\t"
        "vfmul.vf     v24, v8, f12            \n\t"
        "vfmul.vf     v28, v12, f13           \n\t"
        "vfcvt.x.f.v  v16, v16                \n\t"
        "vfcvt.x.f.v  v20, v20                \n\t"
        "vfcvt.x.f.v  v24, v24                \n\t"
        "vfcvt.x.f.v  v28, v28                \n\t"
        "vsetvli      t0, zero, e16, m2       \n\t"
        "vnclip.wx    v16, v16, zero          \n\t"
        "vnclip.wx    v20, v20, zero          \n\t"
        "vnclip.wx    v24, v24, zero          \n\t"
        "vnclip.wx    v28, v28, zero          \n\t"
        "vsetvli      t0, t1, e8, m1          \n\t"
        "vnclip.wx    v16, v16, zero          \n\t"
        "vnclip.wx    v20, v20, zero          \n\t"
        "vnclip.wx    v24, v24, zero          \n\t"
        "vnclip.wx    v28, v28, zero          \n\t"
        "vse8.v       v16, (s1)               \n\t"
        "addi         s1, s1, 140             \n\t"
        "vse8.v       v20, (s2)               \n\t"
        "addi         s2, s2, 140             \n\t"
        "vse8.v       v24, (s3)               \n\t"
        "addi         s3, s3, 140             \n\t"
        "vse8.v       v28, (s4)               \n\t"
        "addi         s4, s4, 140             \n\t"
        "bge          %[K], t3, LOOP_MAIN%=   \n\t"
        "blt          %[K], t2, TAIL%=        \n\t"
        "LOOP_K%=:                            \n\t"
        "vsetvli      t1, %[K], e32, m4       \n\t"
        "vle32.v      v0, (a1)                \n\t"
        "addi         a1, a1, 128             \n\t"
        "sub          %[K], %[K], t1          \n\t"
        "vfabs.v      v16, v0                 \n\t"
        "vsetvli      t0, zero, e32, m2       \n\t"
        "vfmax.vv     v16, v16, v18           \n\t"
        "vsetvli      t0, zero, e32, m1       \n\t"
        "vfmax.vv     v16, v16, v17           \n\t"
        "vfredmax.vs  v17, v16, v17           \n\t"
        "vfmv.f.s     f10,  v17               \n\t"

        "fmul.s       f10, f10, %[RMAXREC]    \n\t"
        "fsw          f10, (s1)               \n\t"
        "addi         s1, s1, 4               \n\t"
        "fdiv.s       f11, %[FONE], f10       \n\t"
        "vsetvli      t0, zero, e32, m4       \n\t"
        "vfmul.vf     v16, v0, f11            \n\t"
        "vfcvt.x.f.v  v16, v16                \n\t"
        "vsetvli      t0, zero, e16, m2       \n\t"
        "vnclip.wx    v16, v16, zero          \n\t"
        "vsetvli      t0, zero, e8, m1        \n\t"
        "vnclip.wx    v16, v16, zero          \n\t"
        "vse8.v       v16, (s1)               \n\t"
        "addi         s1, s1, 32              \n\t"
        "bge          %[K], t2, LOOP_K%=      \n\t"
        "TAIL%=:                              \n\t"
        "blez         %[K], END%=             \n\t"
        "vsetvli      t0, t3, e32, m4         \n\t"
        "vxor.vv      v0, v0, v0              \n\t"
        "vxor.vv      v16, v16, v16           \n\t"
        "jal          x0, LOOP_K%=            \n\t"
        "END%=:                               \n\t"
        : [K] "+r"(CountK)
        : [FONE] "f"(fone), [RMAXREC] "f"(range_max_reciprocal), [SRC] "r"(SRC), [DST] "r"(DST)
        : "cc", "t3", "t2", "t1", "t0", "a1", "a2", "a3", "a4", "s1", "s2", "s3", "s4", "f10", "f11", "f12", "f13");
}

}  // namespace ime1

namespace {
#define SQ4BIT_KERNEL_COMP_1x8x2_4X8X4          \
    "vmadot       v16, v14, v0            \n\t" \
    "vmadot       v18, v14, v1            \n\t" \
    "vmadot       v20, v14, v2            \n\t" \
    "vmadot       v22, v14, v3            \n\t" \
    "vmadot       v16, v15, v4            \n\t" \
    "vmadot       v18, v15, v5            \n\t" \
    "vmadot       v20, v15, v6            \n\t" \
    "vmadot       v22, v15, v7            \n\t"

#define SQ4BIT_KERNEL_ACC_1X4X4                 \
    "vfcvt.f.x.v  v16,  v16               \n\t" \
    "vfcvt.f.x.v  v18,  v18               \n\t" \
    "vfcvt.f.x.v  v20,  v20               \n\t" \
    "vfcvt.f.x.v  v22,  v22               \n\t" \
    "addi         s2, s1, 16              \n\t" \
    "addi         s3, s1, 32              \n\t" \
    "addi         s4, s1, 48              \n\t" \
    "addi         s6, s5, 12              \n\t" \
    "vfmacc.vv    v28, v16, v24           \n\t" \
    "vfmacc.vv    v29, v18, v25           \n\t" \
    "vfmacc.vv    v30, v20, v26           \n\t" \
    "vfmacc.vv    v31, v22, v27           \n\t"

#define SQ4BIT_KERNEL_ACC_F16_1X4X4             \
    "vfcvt.f.x.v  v16,  v16               \n\t" \
    "vfcvt.f.x.v  v18,  v18               \n\t" \
    "vfcvt.f.x.v  v20,  v20               \n\t" \
    "vfcvt.f.x.v  v22,  v22               \n\t" \
    "addi         s2, s1, 8               \n\t" \
    "addi         s3, s1, 16              \n\t" \
    "addi         s4, s1, 24              \n\t" \
    "addi         s6, s5, 12              \n\t" \
    "vfmacc.vv    v28, v16, v24           \n\t" \
    "vfmacc.vv    v29, v18, v25           \n\t" \
    "vfmacc.vv    v30, v20, v26           \n\t" \
    "vfmacc.vv    v31, v22, v27           \n\t"

#define SQ4BIT_KERNEL_LOAD_1x8x2_4X8X4          \
    "vle8.v       v4, (s1)                \n\t" \
    "addi         s1, s1, 128             \n\t" \
    "vle8.v       v5, (s2)                \n\t" \
    "addi         s2, s2, 128             \n\t" \
    "vle8.v       v6, (s3)                \n\t" \
    "addi         s3, s3, 128             \n\t" \
    "vle8.v       v7, (s4)                \n\t" \
    "addi         s4, s4, 128             \n\t" \
    "vsetvli      t0, zero, e8, mf4       \n\t" \
    "vle8.v       v14, (s5)               \n\t" \
    "addi         s5, s5, 16              \n\t" \
    "vle8.v       v15, (s6)               \n\t" \
    "addi         s6, s6, 16              \n\t" \
    "addi         t5, t5, -1              \n\t" \
    "vsetvli      t0, zero, e8, m1        \n\t" \
    "vand.vi      v0, v4, 15              \n\t" \
    "vand.vi      v1, v5, 15              \n\t" \
    "vand.vi      v2, v6, 15              \n\t" \
    "vand.vi      v3, v7, 15              \n\t" \
    "vsrl.vi      v4, v4, 4               \n\t" \
    "vsrl.vi      v5, v5, 4               \n\t" \
    "vsrl.vi      v6, v6, 4               \n\t" \
    "vsrl.vi      v7, v7, 4               \n\t"

#define SQ4BIT_KERNEL_LOAD_ZP_16X1              \
    "vsetvli      t0, zero, e8, mf2       \n\t" \
    "vle8.v       v1, (s7)                \n\t" \
    "vsetvli      t0, zero, e8, m1        \n\t" \
    "vrgather.vv  v8, v1, v13             \n\t" \
    "vadd.vi      v13, v13, 4             \n\t" \
    "vrgather.vv  v9, v1, v13             \n\t" \
    "vadd.vi      v13, v13, 4             \n\t" \
    "vrgather.vv  v10, v1, v13            \n\t" \
    "vadd.vi      v13, v13, 4             \n\t" \
    "vrgather.vv  v11, v1, v13            \n\t" \
    "vadd.vi      v13, v13, -12           \n\t"

// using for M4Kernel
#define LOAD_B_16x8x2                           \
    "vsetvli      t0, zero, e8, m1        \n\t" \
    "vle8.v       v6, (s1)                \n\t" \
    "addi         s1, s1, 32*4            \n\t" \
    "vle8.v       v7, (s2)                \n\t" \
    "addi         s2, s2, 32*4            \n\t" \
    "vle8.v       v8, (s3)                \n\t" \
    "addi         s3, s3, 32*4            \n\t" \
    "vle8.v       v9, (s4)                \n\t" \
    "addi         s4, s4, 32*4            \n\t" \
                                                \
    "vand.vi      v2, v6, 15              \n\t" \
    "vand.vi      v3, v7, 15              \n\t" \
    "vand.vi      v4, v8, 15              \n\t" \
    "vand.vi      v5, v9, 15              \n\t" \
                                                \
    "vsrl.vi      v6, v6, 4               \n\t" \
    "vsrl.vi      v7, v7, 4               \n\t" \
    "vsrl.vi      v8, v8, 4               \n\t" \
    "vsrl.vi      v9, v9, 4               \n\t"

// [s2|s5, s3, s4, s6]
#define LOAD_SCALE_4x16_FP16                    \
    "addi         s2, s5, -8              \n\t" \
    "addi         s3, s5, 8               \n\t" \
    "addi         s4, s5, 16              \n\t" \
    "addi         s6, s5, 24              \n\t" \
    "li           t1, 0xf0                \n\t" \
    "vmv.s.x      v0, t1                  \n\t" \
    "vsetvli      t0, zero, e16, mf4      \n\t" \
    "vle16.v      v9, (s5)                \n\t" \
    "vle16.v      v11, (s3)               \n\t" \
    "vle16.v      v13, (s4)               \n\t" \
    "vle16.v      v15, (s6)               \n\t" \
    "vsetvli      t0, zero, e16, mf2      \n\t" \
    "vle16.v      v9, (s2), v0.t          \n\t" \
    "vle16.v      v11, (s5), v0.t         \n\t" \
    "vle16.v      v13, (s3), v0.t         \n\t" \
    "vle16.v      v15, (s4), v0.t         \n\t" \
    "vfwcvt.f.f.v v8, v9                  \n\t" \
    "vfwcvt.f.f.v v10, v11                \n\t" \
    "vfwcvt.f.f.v v12, v13                \n\t" \
    "vfwcvt.f.f.v v14, v15                \n\t" \
    "vsetvli      t0, zero, e32, m1       \n\t" \
    "vmv.v.v      v9, v8                  \n\t" \
    "vmv.v.v      v11, v10                \n\t" \
    "vmv.v.v      v13, v12                \n\t" \
    "vmv.v.v      v15, v14                \n\t" \
    "li           t1, 0xf0                \n\t" \
    "vmv.s.x      v0, t1                  \n\t" \
    "vsetvli      t0, zero, e32, mf2      \n\t" \
    "vfmul.vf     v8, v8, f1              \n\t" \
    "vfmul.vf     v10, v10, f1            \n\t" \
    "vfmul.vf     v12, v12, f1            \n\t" \
    "vfmul.vf     v14, v14, f1            \n\t" \
    "vfmul.vf     v9, v9, f3              \n\t" \
    "vfmul.vf     v11, v11, f3            \n\t" \
    "vfmul.vf     v13, v13, f3            \n\t" \
    "vfmul.vf     v15, v15, f3            \n\t" \
    "vsetvli      t0, zero, e32, m1       \n\t" \
    "vfmul.vf     v8, v8, f2, v0.t        \n\t" \
    "vfmul.vf     v10, v10, f2, v0.t      \n\t" \
    "vfmul.vf     v12, v12, f2, v0.t      \n\t" \
    "vfmul.vf     v14, v14, f2, v0.t      \n\t" \
    "vfmul.vf     v9, v9, f4, v0.t        \n\t" \
    "vfmul.vf     v11, v11, f4, v0.t      \n\t" \
    "vfmul.vf     v13, v13, f4, v0.t      \n\t" \
    "vfmul.vf     v15, v15, f4, v0.t      \n\t"

// [Q4_0_K128 / scale-build] Drop-in replacement for LOAD_SCALE_4x16_FP16.
//
// The original writes the A row scales with 8 masked `vfmul.vf` plus 4 `vmv.v.v`
// copies (44 instructions).  Here the two merged A-scale vectors are prebuilt once
// (v2 = [As0 x4 | As1 x4], v3 = [As2 x4 | As3 x4]) and the combined scale becomes
// 8 unmasked `vfmul.vv`, which is 32 instructions for the same maths.
//
// Equivalence: with vle16 mf4 (4 halfwords) the low half of stage 2 is the 4 row
// scales and the lane-4..7 half is self, so vfwcvt gives v4=[Bs0..3 | Bs0..3],
// v5=[Bs4..7 | Bs4..7], and so on.  The vmadot 4x4 sub-tile for rows r..r+3 is pair
// (r/4), so v16..v23 need scales {As0,As2} for pair 0 and {As1,As3} for pair 1 --
// exactly what v2/v3 place in lanes 0..3 / 4..7.  Hence v8..v15 are bit-identical
// to the original and the kernel output is unchanged.
// Clobbers v0, v2..v15 and s2,s3,s4,s6 -- all of which the original macro already
// clobbers, and every s-register here is recomputed at the top of each n-tile pass.
#define LOAD_SCALE_4x16_FP16_OPT                \
    "addi         s2, s5, -8              \n\t" \
    "addi         s3, s5, 8               \n\t" \
    "addi         s4, s5, 16              \n\t" \
    "addi         s6, s5, 24              \n\t" \
    "li           t1, 0xf0                \n\t" \
    "vmv.s.x      v0, t1                  \n\t" \
    "vsetvli      t0, zero, e16, mf4      \n\t" \
    "vle16.v      v9, (s5)                \n\t" \
    "vle16.v      v11, (s3)               \n\t" \
    "vle16.v      v13, (s4)               \n\t" \
    "vle16.v      v15, (s6)               \n\t" \
    "vsetvli      t0, zero, e16, mf2      \n\t" \
    "vle16.v      v9, (s2), v0.t          \n\t" \
    "vle16.v      v11, (s5), v0.t         \n\t" \
    "vle16.v      v13, (s3), v0.t         \n\t" \
    "vle16.v      v15, (s4), v0.t         \n\t" \
    "vfwcvt.f.f.v v4, v9                  \n\t" \
    "vfwcvt.f.f.v v5, v11                 \n\t" \
    "vfwcvt.f.f.v v6, v13                 \n\t" \
    "vfwcvt.f.f.v v7, v15                 \n\t" \
    "vsetvli      t0, zero, e32, m1       \n\t" \
    "vfmv.v.f     v2, f1                  \n\t" \
    "vfmerge.vfm  v2, v2, f2, v0          \n\t" \
    "vfmv.v.f     v3, f3                  \n\t" \
    "vfmerge.vfm  v3, v3, f4, v0          \n\t" \
    "vfmul.vv     v8, v4, v2              \n\t" \
    "vfmul.vv     v9, v4, v3              \n\t" \
    "vfmul.vv     v10, v5, v2             \n\t" \
    "vfmul.vv     v11, v5, v3             \n\t" \
    "vfmul.vv     v12, v6, v2             \n\t" \
    "vfmul.vv     v13, v6, v3             \n\t" \
    "vfmul.vv     v14, v7, v2             \n\t" \
    "vfmul.vv     v15, v7, v3             \n\t"

// [s2|s5, s3, s4, s6]
#define LOAD_SCALE_4x16                         \
    "addi         s2, s5, -16             \n\t" \
    "addi         s3, s5, 16              \n\t" \
    "addi         s4, s5, 32              \n\t" \
    "addi         s6, s5, 48              \n\t" \
    "li           t1, 0xf0                \n\t" \
    "vmv.s.x      v0, t1                  \n\t" \
    "vsetvli      t0, zero, e32, mf2      \n\t" \
    "vle32.v      v8, (s5)                \n\t" \
    "vle32.v      v10, (s3)               \n\t" \
    "vle32.v      v12, (s4)               \n\t" \
    "vle32.v      v14, (s6)               \n\t" \
    "vsetvli      t0, zero, e32, m1       \n\t" \
    "vle32.v      v8, (s2), v0.t          \n\t" \
    "vle32.v      v10, (s5), v0.t         \n\t" \
    "vle32.v      v12, (s3), v0.t         \n\t" \
    "vle32.v      v14, (s4), v0.t         \n\t" \
    "vmv.v.v      v9, v8                  \n\t" \
    "vmv.v.v      v11, v10                \n\t" \
    "vmv.v.v      v13, v12                \n\t" \
    "vmv.v.v      v15, v14                \n\t" \
    "vsetvli      t0, zero, e32, mf2      \n\t" \
    "vfmul.vf     v8, v8, f1              \n\t" \
    "vfmul.vf     v10, v10, f1            \n\t" \
    "vfmul.vf     v12, v12, f1            \n\t" \
    "vfmul.vf     v14, v14, f1            \n\t" \
    "vfmul.vf     v9, v9, f3              \n\t" \
    "vfmul.vf     v11, v11, f3            \n\t" \
    "vfmul.vf     v13, v13, f3            \n\t" \
    "vfmul.vf     v15, v15, f3            \n\t" \
    "vsetvli      t0, zero, e32, m1       \n\t" \
    "vfmul.vf     v8, v8, f2, v0.t        \n\t" \
    "vfmul.vf     v10, v10, f2, v0.t      \n\t" \
    "vfmul.vf     v12, v12, f2, v0.t      \n\t" \
    "vfmul.vf     v14, v14, f2, v0.t      \n\t" \
    "vfmul.vf     v9, v9, f4, v0.t        \n\t" \
    "vfmul.vf     v11, v11, f4, v0.t      \n\t" \
    "vfmul.vf     v13, v13, f4, v0.t      \n\t" \
    "vfmul.vf     v15, v15, f4, v0.t      \n\t"

//[s1| BIAS, s2, s3, s4]
#define LOAD_BIAS                               \
    "vsetvli      t0, zero, e32, mf2      \n\t" \
    "li           t1, 0xf0                \n\t" \
    "vmv.s.x      v0, t1                  \n\t" \
    "addi         s1, %[BIAS], -16        \n\t" \
    "addi         s2, %[BIAS], 16         \n\t" \
    "addi         s3, %[BIAS], 32         \n\t" \
    "addi         s4, %[BIAS], 48         \n\t" \
                                                \
    "vle32.v      v24, (%[BIAS])          \n\t" \
    "vle32.v      v26, (s2)               \n\t" \
    "vle32.v      v28, (s3)               \n\t" \
    "vle32.v      v30, (s4)               \n\t" \
    "vsetvli      t0, zero, e32, m1       \n\t" \
    "vle32.v      v24, (s1), v0.t         \n\t" \
    "vle32.v      v26, (%[BIAS]), v0.t    \n\t" \
    "vle32.v      v28, (s2), v0.t         \n\t" \
    "vle32.v      v30, (s3), v0.t         \n\t" \
    "vmv.v.v      v25, v24                \n\t" \
    "vmv.v.v      v27, v26                \n\t" \
    "vmv.v.v      v29, v28                \n\t" \
    "vmv.v.v      v31, v30                \n\t"

#define SQ4BIT_KERNEL_COMP_4x16x16              \
    "vmadot       v16, v10, v2            \n\t" \
    "vmadot       v18, v10, v3            \n\t" \
    "vmadot       v20, v10, v4            \n\t" \
    "vmadot       v22, v10, v5            \n\t" \
    "vmadot       v16, v11, v6            \n\t" \
    "vmadot       v18, v11, v7            \n\t" \
    "vmadot       v20, v11, v8            \n\t" \
    "vmadot       v22, v11, v9            \n\t"

#define SAVE_RESULT_4x16                        \
    "addi         a1, %[C], 0             \n\t" \
    "add          a2, %[C], %[LDC]        \n\t" \
    "add          a3, a2, %[LDC]          \n\t" \
    "add          a4, a3, %[LDC]          \n\t" \
    "addi         a2, a2, -16             \n\t" \
    "addi         a4, a4, -16             \n\t" \
    "li           t1, 0xf0                \n\t" \
    "vmv.s.x      v0, t1                  \n\t" \
    "vsetvli      t0, zero, e32, mf2      \n\t" \
                                                \
    "vse32.v      v24, (a1)               \n\t" \
    "addi         a1, a1, 16              \n\t" \
    "vse32.v      v25, (a3)               \n\t" \
    "addi         a3, a3, 16              \n\t" \
                                                \
    "vse32.v      v26, (a1)               \n\t" \
    "addi         a1, a1, 16              \n\t" \
    "vse32.v      v27, (a3)               \n\t" \
    "addi         a3, a3, 16              \n\t" \
                                                \
    "vse32.v      v28, (a1)               \n\t" \
    "addi         a1, a1, 16              \n\t" \
    "vse32.v      v29, (a3)               \n\t" \
    "addi         a3, a3, 16              \n\t" \
                                                \
    "vse32.v      v30, (a1)               \n\t" \
    "vse32.v      v31, (a3)               \n\t" \
    "vsetvli      t0, zero, e32, m1       \n\t" \
                                                \
    "vse32.v      v24, (a2), v0.t         \n\t" \
    "addi         a2, a2, 16              \n\t" \
    "vse32.v      v25, (a4), v0.t         \n\t" \
    "addi         a4, a4, 16              \n\t" \
                                                \
    "vse32.v      v26, (a2), v0.t         \n\t" \
    "addi         a2, a2, 16              \n\t" \
    "vse32.v      v27, (a4), v0.t         \n\t" \
    "addi         a4, a4, 16              \n\t" \
                                                \
    "vse32.v      v28, (a2), v0.t         \n\t" \
    "addi         a2, a2, 16              \n\t" \
    "vse32.v      v29, (a4), v0.t         \n\t" \
    "addi         a4, a4, 16              \n\t" \
                                                \
    "vse32.v      v30, (a2), v0.t         \n\t" \
    "vse32.v      v31, (a4), v0.t         \n\t"

#define SQ4BIT_KERNEL_LOAD_ZP_16X1_v2           \
    "vsetvli      t0, zero, e8, mf2       \n\t" \
    "vle8.v       v11, (s6)               \n\t" \
    "vsetvli      t0, zero, e8, m1        \n\t" \
    "vrgather.vv  v12, v11, v1            \n\t" \
    "vadd.vi      v1, v1, 4               \n\t" \
    "vrgather.vv  v13, v11, v1            \n\t" \
    "vadd.vi      v1, v1, 4               \n\t" \
    "vrgather.vv  v14, v11, v1            \n\t" \
    "vadd.vi      v1, v1, 4               \n\t" \
    "vrgather.vv  v15, v11, v1            \n\t" \
    "vadd.vi      v1, v1, -12             \n\t"

template <bool HasZeroPoint, bool FoldOffset = false>
void SQ4BitGemmM4Kernel_CompInt8_ScaleFp16_Impl(size_t          BlkLen,
                                                const uint8_t * QuantA,
                                                const uint8_t * QuantBData,
                                                float *         C,
                                                size_t          CountN,
                                                size_t          BlockCountK,
                                                const size_t    ldc) {
    size_t       LDC   = ldc * sizeof(float);
    const size_t INNER = BlkLen / 16;
    float        tmp[4 * 16];

    if constexpr (HasZeroPoint) {
        for (size_t n = 0; n < CountN; n += 16) {
            size_t    NBLKS         = (CountN - n) > 16 ? 16 : CountN - n;
            uint8_t * QuantBDataPtr = (uint8_t *) QuantBData +             //
                                      n * BlockCountK * BlkLen / 2 +       // b data
                                      n * BlockCountK * sizeof(uint8_t) +  // zp
                                      n * BlockCountK * sizeof(_Float16);  // scale
            float * CPtr = C + n;
            if (NBLKS < 16) {
                CPtr = tmp;
                LDC  = 16 * sizeof(float);
            }

            __asm__ volatile(
                "vsetvli            t0, zero, e32, m8           \n\t"
                "vxor.vv            v24, v24, v24               \n\t"
                "addi               t3, %[BlockCountK], 0       \n\t"
                "vsetvli            t0, zero, e8, m1            \n\t"
                "li                 s1, 24                      \n\t"
                "vmv.v.i            v1, 3                       \n\t"
                "vsetvli            t0, s1, e8, m1              \n\t"
                "vmv.v.i            v1, 2                       \n\t"
                "vsetvli            t0, zero, e8, mf2           \n\t"
                "vmv.v.i            v1, 1                       \n\t"
                "vsetvli            t0, zero, e8, mf4           \n\t"
                "vmv.v.i            v1, 0                       \n\t"
                "addi               a1, %[A], 0                 \n\t"
                "addi               s1, %[B], 0                 \n\t"
                "BLOCK_COUNTK_LOOP%=:                           \n\t"
                // scale offset
                "addi               s5, s1, 0                   \n\t"
                // zp offset
                "addi               s6, s1, 32                  \n\t"
                "addi               s1, s6, 16                  \n\t"
                "addi               s2, s1, 32                  \n\t"
                "addi               s3, s1, 32*2                \n\t"
                "addi               s4, s1, 32*3                \n\t"

                "vsetvli            t0, zero, e32, m8           \n\t"
                "vxor.vv            v16, v16, v16               \n\t"
                // load a scale
                "flw                f1, (a1)                    \n\t"
                "flw                f2, 4(a1)                   \n\t"
                "flw                f3, 8(a1)                   \n\t"
                "flw                f4, 12(a1)                  \n\t"
                "addi               a1, a1, 16                  \n\t"
                "addi               t2, %[INNER], 0             \n\t"

                SQ4BIT_KERNEL_LOAD_ZP_16X1_v2

                "BLOCK_INNER_LOOP%=:                            \n\t"

                LOAD_B_16x8x2

                "vle8.v             v10, (a1)                   \n\t"
                "addi               a1, a1, 32                  \n\t"
                "vle8.v             v11, (a1)                   \n\t"
                "addi               a1, a1, 32                  \n\t"
                "vsub.vv            v2, v2, v12                 \n\t"
                "vsub.vv            v6, v6, v12                 \n\t"
                "vsub.vv            v3, v3, v13                 \n\t"
                "vsub.vv            v7, v7, v13                 \n\t"
                "vsub.vv            v4, v4, v14                 \n\t"
                "vsub.vv            v8, v8, v14                 \n\t"
                "vsub.vv            v5, v5, v15                 \n\t"
                "vsub.vv            v9, v9, v15                 \n\t"

                SQ4BIT_KERNEL_COMP_4x16x16

                "addi               t2, t2, -1                  \n\t"
                "bnez               t2, BLOCK_INNER_LOOP%=      \n\t"

                LOAD_SCALE_4x16_FP16_OPT

                "vsetvli            t0, zero, e32, m8           \n\t"
                "vfcvt.f.x.v        v16, v16                    \n\t"
                "vfmacc.vv          v24, v16, v8                \n\t"
                "addi               t3, t3, -1                  \n\t"
                "bnez               t3, BLOCK_COUNTK_LOOP%=     \n\t"

                "RESULT_SAVE%=:                                 \n\t"

                SAVE_RESULT_4x16

                :
                : [INNER] "r"(INNER), [A] "r"(QuantA), [B] "r"(QuantBDataPtr), [LDC] "r"(LDC),
                  [BlockCountK] "r"(BlockCountK), [C] "r"(CPtr)
                : "cc", "t0", "t1", "t2", "t3", "a1", "a2", "a3", "a4", "f1", "f2", "f3", "f4", "s1", "s2", "s3", "s4",
                  "s5", "s6");
        }
    } else if constexpr (!FoldOffset) {
        for (size_t n = 0; n < CountN; n += 16) {
            size_t    NBLKS         = (CountN - n) > 16 ? 16 : CountN - n;
            uint8_t * QuantBDataPtr = (uint8_t *) QuantBData +             //
                                      n * BlockCountK * BlkLen / 2 +       // b data
                                      n * BlockCountK * sizeof(_Float16);  // scale
            float * CPtr = C + n;
            if (NBLKS < 16) {
                CPtr = tmp;
                LDC  = 16 * sizeof(float);
            }

            __asm__ volatile(
                "vsetvli            t0, zero, e32, m8           \n\t"
                "vxor.vv            v24, v24, v24               \n\t"
                "addi               t3, %[BlockCountK], 0       \n\t"
                "addi               a1, %[A], 0                 \n\t"
                "addi               s1, %[B], 0                 \n\t"
                "vsetvli            t0, zero, e32, m8           \n\t"
                "BLOCK_COUNTK_LOOP%=:                           \n\t"
                "addi               s5, s1, 0                   \n\t"
                "addi               s1, s5, 32                  \n\t"
                "addi               s2, s1, 32                  \n\t"
                "addi               s3, s1, 32*2                \n\t"
                "addi               s4, s1, 32*3                \n\t"
                "vxor.vv            v16, v16, v16               \n\t"
                // load a scale
                "flw                f1, (a1)                    \n\t"
                "flw                f2, 4(a1)                   \n\t"
                "flw                f3, 8(a1)                   \n\t"
                "flw                f4, 12(a1)                  \n\t"
                "addi               a1, a1, 16                  \n\t"
                "addi               t2, %[INNER], 0             \n\t"
                "BLOCK_INNER_LOOP%=:                            \n\t"

                LOAD_B_16x8x2

                "vle8.v             v10, (a1)                   \n\t"
                "addi               a1, a1, 32                  \n\t"
                "vle8.v             v11, (a1)                   \n\t"
                "addi               a1, a1, 32                  \n\t"
                "vadd.vi            v2, v2, -8                  \n\t"
                "vadd.vi            v3, v3, -8                  \n\t"
                "vadd.vi            v4, v4, -8                  \n\t"
                "vadd.vi            v5, v5, -8                  \n\t"
                "vadd.vi            v6, v6, -8                  \n\t"
                "vadd.vi            v7, v7, -8                  \n\t"
                "vadd.vi            v8, v8, -8                  \n\t"
                "vadd.vi            v9, v9, -8                  \n\t"

                SQ4BIT_KERNEL_COMP_4x16x16

                "addi               t2, t2, -1                  \n\t"
                "bnez               t2, BLOCK_INNER_LOOP%=      \n\t"

                LOAD_SCALE_4x16_FP16_OPT

                "vsetvli            t0, zero, e32, m8           \n\t"
                "vfcvt.f.x.v        v16, v16                    \n\t"
                "vfmacc.vv          v24, v16, v8                \n\t"
                "addi               t3, t3, -1                  \n\t"
                "bnez               t3, BLOCK_COUNTK_LOOP%=     \n\t"
                "RESULT_SAVE%=:                                 \n\t"

                SAVE_RESULT_4x16

                :
                : [INNER] "r"(INNER), [A] "r"(QuantA), [B] "r"(QuantBDataPtr), [LDC] "r"(LDC),
                  [BlockCountK] "r"(BlockCountK), [C] "r"(CPtr)
                : "cc", "t0", "t1", "t2", "t3", "a1", "a2", "a3", "a4", "f1", "f2", "f3", "f4", "s1", "s2", "s3", "s4",
                  "s5", "s6");
        }
    } else {
        // [A-offset fold] Same loop as the branch above, with the B zero-point shift moved from
        // "8 vector ops per inner iteration" to "one integer correction per K block":
        //     sum_k a*(q-8) = sum_k a*q - 8*sum_k a
        // `a` is the quantised int8 activation.  quantize_a_4row_i8_with_sum() stores 8*sum_k a
        // per row as the two int32 row-broadcast patterns read below, so the correction is
        // exact integer arithmetic and C is bit-identical to the un-folded kernel.
        for (size_t n = 0; n < CountN; n += 16) {
            size_t    NBLKS         = (CountN - n) > 16 ? 16 : CountN - n;
            uint8_t * QuantBDataPtr = (uint8_t *) QuantBData +             //
                                      n * BlockCountK * BlkLen / 2 +       // b data
                                      n * BlockCountK * sizeof(_Float16);  // scale
            float * CPtr = C + n;
            if (NBLKS < 16) {
                CPtr = tmp;
                LDC  = 16 * sizeof(float);
            }

            __asm__ volatile(
                "vsetvli            t0, zero, e32, m8           \n\t"
                "vxor.vv            v24, v24, v24               \n\t"
                "addi               t3, %[BlockCountK], 0       \n\t"
                "addi               a1, %[A], 0                 \n\t"
                "addi               s1, %[B], 0                 \n\t"
                "vsetvli            t0, zero, e32, m8           \n\t"
                "BLOCK_COUNTK_LOOP%=:                           \n\t"
                "addi               s5, s1, 0                   \n\t"
                "addi               s1, s5, 32                  \n\t"
                "addi               s2, s1, 32                  \n\t"
                "addi               s3, s1, 32*2                \n\t"
                "addi               s4, s1, 32*3                \n\t"
                "vxor.vv            v16, v16, v16               \n\t"
                // load a scale
                "flw                f1, (a1)                    \n\t"
                "flw                f2, 4(a1)                   \n\t"
                "flw                f3, 8(a1)                   \n\t"
                "flw                f4, 12(a1)                  \n\t"
                "addi               a1, a1, 16                  \n\t"
                "addi               t2, %[INNER], 0             \n\t"
                "BLOCK_INNER_LOOP%=:                            \n\t"

                LOAD_B_16x8x2

                "vle8.v             v10, (a1)                   \n\t"
                "addi               a1, a1, 32                  \n\t"
                "vle8.v             v11, (a1)                   \n\t"
                "addi               a1, a1, 32                  \n\t"
                // the 8 per-iteration `vadd.vi -8` live here in the un-folded branch

                SQ4BIT_KERNEL_COMP_4x16x16

                "addi               t2, t2, -1                  \n\t"
                "bnez               t2, BLOCK_INNER_LOOP%=      \n\t"

                // one zero-point correction for the whole K block: a1 now points at the two
                // 32 B row-broadcast patterns that follow this block's 512 B payload
                "vsetvli            t0, zero, e32, m1           \n\t"
                "vle32.v            v15, (a1)                   \n\t"
                "addi               a1, a1, 32                  \n\t"
                "vle32.v            v14, (a1)                   \n\t"
                "addi               a1, a1, 32                  \n\t"
                "vsub.vv            v16, v16, v15               \n\t"
                "vsub.vv            v17, v17, v14               \n\t"
                "vsub.vv            v18, v18, v15               \n\t"
                "vsub.vv            v19, v19, v14               \n\t"
                "vsub.vv            v20, v20, v15               \n\t"
                "vsub.vv            v21, v21, v14               \n\t"
                "vsub.vv            v22, v22, v15               \n\t"
                "vsub.vv            v23, v23, v14               \n\t"

                LOAD_SCALE_4x16_FP16_OPT

                "vsetvli            t0, zero, e32, m8           \n\t"
                "vfcvt.f.x.v        v16, v16                    \n\t"
                "vfmacc.vv          v24, v16, v8                \n\t"
                "addi               t3, t3, -1                  \n\t"
                "bnez               t3, BLOCK_COUNTK_LOOP%=     \n\t"
                "RESULT_SAVE%=:                                 \n\t"

                SAVE_RESULT_4x16

                :
                : [INNER] "r"(INNER), [A] "r"(QuantA), [B] "r"(QuantBDataPtr), [LDC] "r"(LDC),
                  [BlockCountK] "r"(BlockCountK), [C] "r"(CPtr)
                : "cc", "t0", "t1", "t2", "t3", "a1", "a2", "a3", "a4", "f1", "f2", "f3", "f4", "s1", "s2", "s3", "s4",
                  "s5", "s6");
        }
    }
}

// GEMV prefetch lookahead in K-blocks. The M=1 kernel is latency-bound on DDR reads and
// the caller streams the weights straight from DDR (no TCM), so prefetch a few K-blocks
// ahead. Uses plain loads: `prefetch.r` needs Zicbop, which the a60/x60 ISA string does
// not advertise. 0 disables.
static int gemv_pf_blocks() {
    static int pf = [] {
        const char * e = getenv("GGML_SPACEMIT_GEMV_PF");
        int          v = (e != NULL && e[0] != '\0') ? atoi(e) : 0;
        if (v < 0) {
            v = 0;
        }
        if (v > 6) {
            v = 6;
        }
        return v;
    }();
    return pf;
}

template <bool HasZeroPoint>
void SQ4BitGemmM1Kernel_CompInt8_ScaleFp16_Impl(size_t          BlkLen,
                                                const uint8_t * QuantA,
                                                const uint8_t * QuantBData,
                                                float *         C,
                                                size_t          CountN,
                                                size_t          BlockCountK,
                                                const size_t    ldc) {
    GGML_UNUSED(ldc);
    size_t INNER = BlkLen / 16;
    const int pf = gemv_pf_blocks();

    if constexpr (HasZeroPoint) {
        for (size_t n = 0; n < CountN; n += 16) {
            size_t    nblks         = (CountN - n) > 16 ? 16 : CountN - n;
            uint8_t * QuantBDataPtr = (uint8_t *) QuantBData +             //
                                      n * BlockCountK * BlkLen / 2 +       // b data
                                      n * BlockCountK * sizeof(uint8_t) +  // zp
                                      n * BlockCountK * sizeof(_Float16);  // scale
            float * CPtr = C + n;
            size_t  cnt  = BlockCountK;

            __asm__ volatile(
                "vsetvli      t0, zero, e32, m4       \n\t"
                "vxor.vv      v28, v28, v28           \n\t"

                "vsetvli      t0, zero, e8, m1        \n\t"
                "vmv.v.i      v13, 3                  \n\t"
                "li           s1, 24                  \n\t"
                "vsetvli      t0, s1, e8, m1          \n\t"
                "vmv.v.i      v13, 2                  \n\t"
                "vsetvli      t0, zero, e8, mf2       \n\t"
                "vmv.v.i      v13, 1                  \n\t"
                "vsetvli      t0, zero, e8, mf4       \n\t"
                "vmv.v.i      v13, 0                  \n\t"

                "addi         s1, %[B], 0             \n\t"
                "addi         s2, %[B], 8             \n\t"
                "addi         s3, %[B], 16            \n\t"
                "addi         s4, %[B], 24            \n\t"

                "addi         s7, %[B], 32            \n\t"

                "addi         s5, %[A], 0             \n\t"
                "addi         s6, %[A], 12            \n\t"
                "li           t2, 304                 \n\t"
                "mul          t2, %[PF], t2           \n\t"
                "add          t4, %[B], t2            \n\t"
                "LOOP_K%=:                            \n\t"
                "vsetvli      t0, zero, e16, mf4      \n\t"
                "vle16.v      v4, (s1)                \n\t"
                "addi         s1, s1, 48              \n\t"
                "vle16.v      v5, (s2)                \n\t"
                "addi         s2, s2, 72              \n\t"
                "vle16.v      v6, (s3)                \n\t"
                "addi         s3, s3, 96              \n\t"
                "vle16.v      v7, (s4)                \n\t"
                "addi         s4, s4, 120             \n\t"
                "flw          f1, (s5)                \n\t"
                "addi         s5, s5, 4               \n\t"
                "beqz         %[PF], PFDONE%=          \n\t"
                "blt          %[PF], %[CNT], PFOK%=   \n\t"
                "jal          x0, PFDONE%=            \n\t"
                "PFOK%=:                              \n\t"
                "ld           t2, 0(t4)               \n\t"
                "ld           t2, 64(t4)              \n\t"
                "ld           t2, 128(t4)             \n\t"
                "ld           t2, 192(t4)             \n\t"
                "ld           t2, 256(t4)             \n\t"
                "ld           t2, 296(t4)             \n\t"
                "PFDONE%=:                            \n\t"

                "vfwcvt.f.f.v v8, v4                  \n\t"
                "vfwcvt.f.f.v v9, v5                  \n\t"
                "vfwcvt.f.f.v v10, v6                 \n\t"
                "vfwcvt.f.f.v v11, v7                 \n\t"
                "vsetvli      t0, zero, e32, mf2      \n\t"

                "addi         t5, %[INNER], 0         \n\t"
                "vxor.vv      v16, v16, v16           \n\t"
                "vxor.vv      v18, v18, v18           \n\t"
                "vxor.vv      v20, v20, v20           \n\t"
                "vxor.vv      v22, v22, v22           \n\t"
                "vfmul.vf     v24, v8, f1             \n\t"
                "vfmul.vf     v25, v9, f1             \n\t"
                "vfmul.vf     v26, v10, f1            \n\t"
                "vfmul.vf     v27, v11, f1            \n\t"
                "addi         %[CNT], %[CNT], -1      \n\t"

                SQ4BIT_KERNEL_LOAD_ZP_16X1

                "LOOP_INNER%=:                        \n\t"

                SQ4BIT_KERNEL_LOAD_1x8x2_4X8X4

                "vsub.vv      v0, v0, v8              \n\t"
                "vsub.vv      v4, v4, v8              \n\t"
                "vsub.vv      v1, v1, v9              \n\t"
                "vsub.vv      v5, v5, v9              \n\t"
                "vsub.vv      v2, v2, v10             \n\t"
                "vsub.vv      v6, v6, v10             \n\t"
                "vsub.vv      v3, v3, v11             \n\t"
                "vsub.vv      v7, v7, v11             \n\t"

                SQ4BIT_KERNEL_COMP_1x8x2_4X8X4

                "bnez         t5, LOOP_INNER%=        \n\t"
                "vsetvli      t0, zero, e32, mf2      \n\t"

                SQ4BIT_KERNEL_ACC_F16_1X4X4
                "addi         s7, s1, 32              \n\t"
                "addi         t4, t4, 304             \n\t"

                "bnez         %[CNT], LOOP_K%=        \n\t"
                "addi         t3, zero, 16            \n\t"
                "addi         s1, %[C], 16            \n\t"
                "addi         s2, %[C], 32            \n\t"
                "addi         s3, %[C], 48            \n\t"
                "blt          %[NBLKS], t3, ST_TAIL%= \n\t"
                "vse32.v      v28, (%[C])             \n\t"
                "vse32.v      v29, (s1)               \n\t"
                "vse32.v      v30, (s2)               \n\t"
                "vse32.v      v31, (s3)               \n\t"
                "jal          x0, END%=               \n\t"

                "ST_TAIL%=:                           \n\t"
                "vsetvli      t0, %[NBLKS], e32, mf2  \n\t"
                "sub          %[NBLKS], %[NBLKS], t0  \n\t"
                "vse32.v      v28, (%[C])             \n\t"
                "vsetvli      t0, %[NBLKS], e32, mf2  \n\t"
                "sub          %[NBLKS], %[NBLKS], t0  \n\t"
                "vse32.v      v29, (s1)               \n\t"
                "vsetvli      t0, %[NBLKS], e32, mf2  \n\t"
                "sub          %[NBLKS], %[NBLKS], t0  \n\t"
                "vse32.v      v30, (s2)               \n\t"
                "vsetvli      t0, %[NBLKS], e32, mf2  \n\t"
                "sub          %[NBLKS], %[NBLKS], t0  \n\t"
                "vse32.v      v31, (s3)               \n\t"
                "END%=:                               \n\t"

                : [CNT] "+r"(cnt), [NBLKS] "+r"(nblks)
                : [INNER] "r"(INNER), [A] "r"(QuantA), [B] "r"(QuantBDataPtr), [C] "r"(CPtr), [PF] "r"(pf)
                : "cc", "t0", "t5", "t3", "t2", "t4", "f1", "s1", "s2", "s3", "s4", "s5", "s6", "s7");
        }
    } else {
        for (size_t n = 0; n < CountN; n += 16) {
            size_t    nblks         = (CountN - n) > 16 ? 16 : CountN - n;
            uint8_t * QuantBDataPtr = (uint8_t *) QuantBData +             //
                                      n * BlockCountK * BlkLen / 2 +       // b data
                                      n * BlockCountK * sizeof(_Float16);  // scale
            float * CPtr = C + n;
            size_t  cnt  = BlockCountK;

            __asm__ volatile(
                "vsetvli      t0, zero, e32, m4       \n\t"
                "vxor.vv      v28, v28, v28           \n\t"
                "addi         s1, %[B], 0             \n\t"
                "addi         s2, %[B], 8             \n\t"
                "addi         s3, %[B], 16            \n\t"
                "addi         s4, %[B], 24            \n\t"

                "addi         s5, %[A], 0             \n\t"
                "addi         s6, %[A], 12            \n\t"
                "li           t2, 288                 \n\t"
                "mul          t2, %[PF], t2           \n\t"
                "add          t4, %[B], t2            \n\t"
                "LOOP_K%=:                            \n\t"
                "vsetvli      t0, zero, e16, mf4      \n\t"
                "vle16.v      v4, (s1)                \n\t"
                "addi         s1, s1, 32              \n\t"
                "vle16.v      v5, (s2)                \n\t"
                "addi         s2, s2, 56              \n\t"
                "vle16.v      v6, (s3)                \n\t"
                "addi         s3, s3, 80              \n\t"
                "vle16.v      v7, (s4)                \n\t"
                "addi         s4, s4, 104             \n\t"
                "flw          f1, (s5)                \n\t"
                "addi         s5, s5, 4               \n\t"
                "beqz         %[PF], PFDONE%=          \n\t"
                "blt          %[PF], %[CNT], PFOK%=   \n\t"
                "jal          x0, PFDONE%=            \n\t"
                "PFOK%=:                              \n\t"
                "ld           t2, 0(t4)               \n\t"
                "ld           t2, 64(t4)              \n\t"
                "ld           t2, 128(t4)             \n\t"
                "ld           t2, 192(t4)             \n\t"
                "ld           t2, 256(t4)             \n\t"
                "PFDONE%=:                            \n\t"

                "vfwcvt.f.f.v v8, v4                  \n\t"
                "vfwcvt.f.f.v v9, v5                  \n\t"
                "vfwcvt.f.f.v v10, v6                 \n\t"
                "vfwcvt.f.f.v v11, v7                 \n\t"
                "vsetvli      t0, zero, e32, mf2      \n\t"

                "addi         t5, %[INNER], 0         \n\t"
                "vxor.vv      v16, v16, v16           \n\t"
                "vxor.vv      v18, v18, v18           \n\t"
                "vxor.vv      v20, v20, v20           \n\t"
                "vxor.vv      v22, v22, v22           \n\t"
                "vfmul.vf     v24, v8, f1             \n\t"
                "vfmul.vf     v25, v9, f1             \n\t"
                "vfmul.vf     v26, v10, f1            \n\t"
                "vfmul.vf     v27, v11, f1            \n\t"
                "addi         %[CNT], %[CNT], -1      \n\t"
                "vsetvli      t0, zero, e8, m1        \n\t"
                "LOOP_INNER%=:                        \n\t"

                SQ4BIT_KERNEL_LOAD_1x8x2_4X8X4

                "vadd.vi      v0, v0, -8              \n\t"
                "vadd.vi      v1, v1, -8              \n\t"
                "vadd.vi      v2, v2, -8              \n\t"
                "vadd.vi      v3, v3, -8              \n\t"
                "vadd.vi      v4, v4, -8              \n\t"
                "vadd.vi      v5, v5, -8              \n\t"
                "vadd.vi      v6, v6, -8              \n\t"
                "vadd.vi      v7, v7, -8              \n\t"

                SQ4BIT_KERNEL_COMP_1x8x2_4X8X4

                "bnez         t5, LOOP_INNER%=        \n\t"
                "vsetvli      t0, zero, e32, mf2      \n\t"

                SQ4BIT_KERNEL_ACC_F16_1X4X4
                "addi         t4, t4, 288             \n\t"

                "bnez         %[CNT], LOOP_K%=        \n\t"
                "addi         t3, zero, 16            \n\t"
                "addi         s1, %[C], 16            \n\t"
                "addi         s2, %[C], 32            \n\t"
                "addi         s3, %[C], 48            \n\t"
                "blt          %[NBLKS], t3, ST_TAIL%= \n\t"
                "vse32.v      v28, (%[C])             \n\t"
                "vse32.v      v29, (s1)               \n\t"
                "vse32.v      v30, (s2)               \n\t"
                "vse32.v      v31, (s3)               \n\t"
                "jal          x0, END%=               \n\t"

                "ST_TAIL%=:                           \n\t"
                "vsetvli      t0, %[NBLKS], e32, mf2  \n\t"
                "sub          %[NBLKS], %[NBLKS], t0  \n\t"
                "vse32.v      v28, (%[C])             \n\t"
                "vsetvli      t0, %[NBLKS], e32, mf2  \n\t"
                "sub          %[NBLKS], %[NBLKS], t0  \n\t"
                "vse32.v      v29, (s1)               \n\t"
                "vsetvli      t0, %[NBLKS], e32, mf2  \n\t"
                "sub          %[NBLKS], %[NBLKS], t0  \n\t"
                "vse32.v      v30, (s2)               \n\t"
                "vsetvli      t0, %[NBLKS], e32, mf2  \n\t"
                "sub          %[NBLKS], %[NBLKS], t0  \n\t"
                "vse32.v      v31, (s3)               \n\t"
                "END%=:                               \n\t"

                : [CNT] "+r"(cnt), [NBLKS] "+r"(nblks)
                : [INNER] "r"(INNER), [A] "r"(QuantA), [B] "r"(QuantBDataPtr), [C] "r"(CPtr), [PF] "r"(pf)
                : "cc", "t0", "t5", "t3", "t2", "t4", "f1", "s1", "s2", "s3", "s4", "s5", "s6");
        }
    }
}

}  // namespace



namespace ime1 {
size_t gemm_kernel_i8i4(size_t          blk_len,
                        const uint8_t * quant_a_ptr,
                        const uint8_t * quant_b_data,
                        const uint8_t * quant_b_zp,
                        float *         c_ptr,
                        size_t          count_m,
                        size_t          count_n,
                        size_t          k_blks,
                        size_t          ldc) {
    if (count_m >= 4) {
        if (quant_b_zp != nullptr) {
            SQ4BitGemmM4Kernel_CompInt8_ScaleFp16_Impl<true>(blk_len, quant_a_ptr, quant_b_data, c_ptr, count_n, k_blks,
                                                             ldc);
        } else {
            SQ4BitGemmM4Kernel_CompInt8_ScaleFp16_Impl<false>(blk_len, quant_a_ptr, quant_b_data, c_ptr, count_n,
                                                              k_blks, ldc);
        }
        return 4;
    } else {
        if (quant_b_zp != nullptr) {
            SQ4BitGemmM1Kernel_CompInt8_ScaleFp16_Impl<true>(blk_len, quant_a_ptr, quant_b_data, c_ptr, count_n, k_blks,
                                                             ldc);
        } else {
            SQ4BitGemmM1Kernel_CompInt8_ScaleFp16_Impl<false>(blk_len, quant_a_ptr, quant_b_data, c_ptr, count_n,
                                                              k_blks, ldc);
        }
        return 1;
    }
}

// [A-offset fold] Entry point for the folded kernel.  Only the BlkLen=128 / count_m>=4 /
// no-zero-point path has the folded correction; anything else falls back to gemm_kernel_i8i4
// (ime.cpp only selects this when the A operand came from quantize_a_4row_i8_with_sum).
size_t gemm_kernel_i8i4_fold(size_t          blk_len,
                             const uint8_t * quant_a_ptr,
                             const uint8_t * quant_b_data,
                             const uint8_t * quant_b_zp,
                             float *         c_ptr,
                             size_t          count_m,
                             size_t          count_n,
                             size_t          k_blks,
                             size_t          ldc) {
    if (blk_len != 128 || count_m < 4 || quant_b_zp != nullptr) {
        return gemm_kernel_i8i4(blk_len, quant_a_ptr, quant_b_data, quant_b_zp, c_ptr, count_m, count_n, k_blks, ldc);
    }
    SQ4BitGemmM4Kernel_CompInt8_ScaleFp16_Impl<false, true>(blk_len, quant_a_ptr, quant_b_data, c_ptr, count_n, k_blks,
                                                            ldc);
    return 4;
}

// ----------------------------------------------------------------------------
// int8 x int8 GEMM for Q8_0 weights (block_q8_0x16 interleaved repack, no dequant).
// Activation layout per K-block (BlkLen=32): [4 f32 scales (16B)][K-tile0..3],
// each K-tile = 4 rows x 8 K (144 B/K-block). Single row: [f32 scale (4B)][32 int8].
// Weight layout per K-block: [16 fp16 scales (32B)][K-tile0..3], each K-tile =
// 16 cols x 8 K (so a 4-col N-tile = 32 contiguous int8 = B^T for the vmadot).
// A60 vmadot tile = M=4,N=4,K=8; vs1 LMUL selects M (e8 m1 -> M=4, e8 mf4 -> M=1).
// ----------------------------------------------------------------------------
static void SQ8BitGemmM4Kernel_CompInt8_ScaleFp32_Impl(size_t BlkLen, const uint8_t *QuantA, const uint8_t *QuantBData, float *C, size_t CountN,
                               size_t BlockCountK, size_t ldc) {
    const size_t b_stride = BlockCountK * (BlkLen + sizeof(_Float16));  // per column

    for (size_t n = 0; n < CountN; n += 16) {
        const uint8_t * b_ptr = QuantBData + n * b_stride;
        float *         c_ptr = C + n;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 16; j++) c_ptr[i * ldc + j] = 0.0f;
        }

        int32_t tmp[64];
        for (size_t k = 0; k < BlockCountK; k++) {
            const float *  a_scale = (const float *) (QuantA + k * 144);
            const uint8_t *a_int8  = QuantA + k * 144 + 16;
            const uint8_t *b_int8  = b_ptr + 32;

            __asm__ volatile(
                "addi            a1, %[A], 0                \n\t"
                "addi            s5, %[B], 0                \n\t"
                "vsetvli         t0, zero, e32, m2          \n\t"
                "vxor.vv         v16, v16, v16              \n\t"
                "vxor.vv         v18, v18, v18              \n\t"
                "vxor.vv         v20, v20, v20              \n\t"
                "vxor.vv         v22, v22, v22              \n\t"
                // K-tile 0
                "vsetvli         t0, zero, e8, m1           \n\t"
                "vle8.v          v0, (a1)                   \n\t"
                "addi            a1, a1, 32                 \n\t"
                "vle8.v          v2, (s5)                   \n\t"
                "addi            s5, s5, 32                 \n\t"
                "vle8.v          v4, (s5)                   \n\t"
                "addi            s5, s5, 32                 \n\t"
                "vle8.v          v6, (s5)                   \n\t"
                "addi            s5, s5, 32                 \n\t"
                "vle8.v          v8, (s5)                   \n\t"
                "addi            s5, s5, 32                 \n\t"
                "vmadot          v16, v0, v2                \n\t"
                "vmadot          v18, v0, v4                \n\t"
                "vmadot          v20, v0, v6                \n\t"
                "vmadot          v22, v0, v8                \n\t"
                // K-tile 1
                "vle8.v          v0, (a1)                   \n\t"
                "addi            a1, a1, 32                 \n\t"
                "vle8.v          v2, (s5)                   \n\t"
                "addi            s5, s5, 32                 \n\t"
                "vle8.v          v4, (s5)                   \n\t"
                "addi            s5, s5, 32                 \n\t"
                "vle8.v          v6, (s5)                   \n\t"
                "addi            s5, s5, 32                 \n\t"
                "vle8.v          v8, (s5)                   \n\t"
                "addi            s5, s5, 32                 \n\t"
                "vmadot          v16, v0, v2                \n\t"
                "vmadot          v18, v0, v4                \n\t"
                "vmadot          v20, v0, v6                \n\t"
                "vmadot          v22, v0, v8                \n\t"
                // K-tile 2
                "vle8.v          v0, (a1)                   \n\t"
                "addi            a1, a1, 32                 \n\t"
                "vle8.v          v2, (s5)                   \n\t"
                "addi            s5, s5, 32                 \n\t"
                "vle8.v          v4, (s5)                   \n\t"
                "addi            s5, s5, 32                 \n\t"
                "vle8.v          v6, (s5)                   \n\t"
                "addi            s5, s5, 32                 \n\t"
                "vle8.v          v8, (s5)                   \n\t"
                "addi            s5, s5, 32                 \n\t"
                "vmadot          v16, v0, v2                \n\t"
                "vmadot          v18, v0, v4                \n\t"
                "vmadot          v20, v0, v6                \n\t"
                "vmadot          v22, v0, v8                \n\t"
                // K-tile 3
                "vle8.v          v0, (a1)                   \n\t"
                "addi            a1, a1, 32                 \n\t"
                "vle8.v          v2, (s5)                   \n\t"
                "addi            s5, s5, 32                 \n\t"
                "vle8.v          v4, (s5)                   \n\t"
                "addi            s5, s5, 32                 \n\t"
                "vle8.v          v6, (s5)                   \n\t"
                "addi            s5, s5, 32                 \n\t"
                "vle8.v          v8, (s5)                   \n\t"
                "addi            s5, s5, 32                 \n\t"
                "vmadot          v16, v0, v2                \n\t"
                "vmadot          v18, v0, v4                \n\t"
                "vmadot          v20, v0, v6                \n\t"
                "vmadot          v22, v0, v8                \n\t"

                "vsetvli         t0, zero, e32, m2          \n\t"
                "vse32.v         v16, (%[TMP])              \n\t"
                "addi            t2, %[TMP], 64             \n\t"
                "vse32.v         v18, (t2)                  \n\t"
                "addi            t2, t2, 64                 \n\t"
                "vse32.v         v20, (t2)                  \n\t"
                "addi            t2, t2, 64                 \n\t"
                "vse32.v         v22, (t2)                  \n\t"
                :
                : [A] "r"(a_int8), [B] "r"(b_int8), [TMP] "r"(tmp)
                : "cc", "t0", "t1", "t2", "a1", "s5", "memory");
            {
                float b_scale[16];
                for (int j = 0; j < 16; j++) b_scale[j] = (float) ((const _Float16 *) b_ptr)[j];
                for (int i = 0; i < 4; i++) {
                    const float as = a_scale[i];
                    for (int tile = 0; tile < 4; tile++) {
                        for (int j = 0; j < 4; j++) {
                            c_ptr[i * ldc + tile * 4 + j] += (float) tmp[tile * 16 + i * 4 + j] * as * b_scale[tile * 4 + j];
                        }
                    }
                }
            }
            b_ptr += BlkLen * 16 + sizeof(_Float16) * 16;
        }
    }
}

static void SQ8BitGemmM1Kernel_CompInt8_ScaleFp32_Impl(size_t BlkLen, const uint8_t *QuantA, const uint8_t *QuantBData, float *C, size_t CountN,
                               size_t BlockCountK, size_t ldc) {
    GGML_UNUSED(ldc);
    const size_t b_stride = BlockCountK * (BlkLen + sizeof(_Float16));

    for (size_t n = 0; n < CountN; n += 16) {
        const uint8_t * b_ptr = QuantBData + n * b_stride;
        float *         c_ptr = C + n;
        float           acc[16];
        for (int j = 0; j < 16; j++) acc[j] = 0.0f;

        int32_t tmp[16];
        for (size_t k = 0; k < BlockCountK; k++) {
            const float   a_scale = *(const float *) (QuantA + k * 36);
            const uint8_t *a_int8 = QuantA + k * 36 + 4;
            const uint8_t *b_int8 = b_ptr + 32;

            __asm__ volatile(
                "addi            a1, %[A], 0                \n\t"
                "addi            s5, %[B], 0                \n\t"
                "vsetvli         t0, zero, e32, mf2         \n\t"
                "vxor.vv         v16, v16, v16              \n\t"
                "vxor.vv         v18, v18, v18              \n\t"
                "vxor.vv         v20, v20, v20              \n\t"
                "vxor.vv         v22, v22, v22              \n\t"
                // K-tile 0
                "vsetvli         t0, zero, e8, mf4          \n\t"
                "vle8.v          v14, (a1)                  \n\t"
                "addi            a1, a1, 8                  \n\t"
                "vsetvli         t0, zero, e8, m1           \n\t"
                "vle8.v          v0, (s5)                   \n\t"
                "addi            s5, s5, 32                 \n\t"
                "vle8.v          v1, (s5)                   \n\t"
                "addi            s5, s5, 32                 \n\t"
                "vle8.v          v2, (s5)                   \n\t"
                "addi            s5, s5, 32                 \n\t"
                "vle8.v          v3, (s5)                   \n\t"
                "addi            s5, s5, 32                 \n\t"
                "vmadot          v16, v14, v0               \n\t"
                "vmadot          v18, v14, v1               \n\t"
                "vmadot          v20, v14, v2               \n\t"
                "vmadot          v22, v14, v3               \n\t"
                // K-tile 1
                "vsetvli         t0, zero, e8, mf4          \n\t"
                "vle8.v          v14, (a1)                  \n\t"
                "addi            a1, a1, 8                  \n\t"
                "vsetvli         t0, zero, e8, m1           \n\t"
                "vle8.v          v0, (s5)                   \n\t"
                "addi            s5, s5, 32                 \n\t"
                "vle8.v          v1, (s5)                   \n\t"
                "addi            s5, s5, 32                 \n\t"
                "vle8.v          v2, (s5)                   \n\t"
                "addi            s5, s5, 32                 \n\t"
                "vle8.v          v3, (s5)                   \n\t"
                "addi            s5, s5, 32                 \n\t"
                "vmadot          v16, v14, v0               \n\t"
                "vmadot          v18, v14, v1               \n\t"
                "vmadot          v20, v14, v2               \n\t"
                "vmadot          v22, v14, v3               \n\t"
                // K-tile 2
                "vsetvli         t0, zero, e8, mf4          \n\t"
                "vle8.v          v14, (a1)                  \n\t"
                "addi            a1, a1, 8                  \n\t"
                "vsetvli         t0, zero, e8, m1           \n\t"
                "vle8.v          v0, (s5)                   \n\t"
                "addi            s5, s5, 32                 \n\t"
                "vle8.v          v1, (s5)                   \n\t"
                "addi            s5, s5, 32                 \n\t"
                "vle8.v          v2, (s5)                   \n\t"
                "addi            s5, s5, 32                 \n\t"
                "vle8.v          v3, (s5)                   \n\t"
                "addi            s5, s5, 32                 \n\t"
                "vmadot          v16, v14, v0               \n\t"
                "vmadot          v18, v14, v1               \n\t"
                "vmadot          v20, v14, v2               \n\t"
                "vmadot          v22, v14, v3               \n\t"
                // K-tile 3
                "vsetvli         t0, zero, e8, mf4          \n\t"
                "vle8.v          v14, (a1)                  \n\t"
                "addi            a1, a1, 8                  \n\t"
                "vsetvli         t0, zero, e8, m1           \n\t"
                "vle8.v          v0, (s5)                   \n\t"
                "addi            s5, s5, 32                 \n\t"
                "vle8.v          v1, (s5)                   \n\t"
                "addi            s5, s5, 32                 \n\t"
                "vle8.v          v2, (s5)                   \n\t"
                "addi            s5, s5, 32                 \n\t"
                "vle8.v          v3, (s5)                   \n\t"
                "addi            s5, s5, 32                 \n\t"
                "vmadot          v16, v14, v0               \n\t"
                "vmadot          v18, v14, v1               \n\t"
                "vmadot          v20, v14, v2               \n\t"
                "vmadot          v22, v14, v3               \n\t"

                "vsetvli         t0, zero, e32, mf2         \n\t"
                "vse32.v         v16, (%[TMP])              \n\t"
                "addi            t2, %[TMP], 16             \n\t"
                "vse32.v         v18, (t2)                  \n\t"
                "addi            t2, t2, 16                 \n\t"
                "vse32.v         v20, (t2)                  \n\t"
                "addi            t2, t2, 16                 \n\t"
                "vse32.v         v22, (t2)                  \n\t"
                :
                : [A] "r"(a_int8), [B] "r"(b_int8), [TMP] "r"(tmp)
                : "cc", "t0", "t1", "t2", "a1", "s5", "memory");
            {
                float b_scale[16];
                for (int j = 0; j < 16; j++) b_scale[j] = (float) ((const _Float16 *) b_ptr)[j];
                for (int tile = 0; tile < 4; tile++) {
                    for (int j = 0; j < 4; j++) {
                        acc[tile * 4 + j] += (float) tmp[tile * 4 + j] * a_scale * b_scale[tile * 4 + j];
                    }
                }
            }
            b_ptr += BlkLen * 16 + sizeof(_Float16) * 16;
        }
        for (int j = 0; j < 16; j++) c_ptr[j] = acc[j];
    }
}



size_t gemm_kernel_i8i8(size_t blk_len, const uint8_t * quant_a_ptr, const uint8_t * quant_b_data, const uint8_t * quant_b_zp,
                        float * c_ptr, size_t count_m, size_t count_n, size_t k_blks, size_t ldc) {
    GGML_UNUSED(quant_b_zp);
    if (count_m >= 4) {
        SQ8BitGemmM4Kernel_CompInt8_ScaleFp32_Impl(blk_len, quant_a_ptr, quant_b_data, c_ptr, count_n, k_blks, ldc);
        return 4;
    } else {
        SQ8BitGemmM1Kernel_CompInt8_ScaleFp32_Impl(blk_len, quant_a_ptr, quant_b_data, c_ptr, count_n, k_blks, ldc);
        return 1;
    }
}

}  // namespace ime1
}  // namespace spacemit_kernels
