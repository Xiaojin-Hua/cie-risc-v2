#define GGML_COMMON_IMPL_CPP
#define GGML_COMMON_DECL_CPP

#include "ime.h"

#include "binary-ops.h"
#include "common.h"
#include "ggml-backend-impl.h"
#include "ggml-common.h"
#include "ggml-cpu.h"
#include "ime_env.h"
#include "ime_kernels.h"
#include "ops.h"
#include "repack.h"
#include "rvv_kernels.h"
#include "spine_mem_pool.h"
#include "traits.h"
#include "vec.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cerrno>
#include <cmath>
#include <cstdio>  // for GGML_ASSERT
#include <cstdlib>  // for getenv (FA probe)
#include <mutex>    // for std::call_once in the A-quantisation cache
#include <stdexcept>
#include <thread>
// clang-format off
#if defined(__riscv)

#if !defined(__riscv_v) || !defined(__riscv_v_intrinsic)
#error "riscv v extension or v_intrinsic not enabled"
#else
#include <riscv_vector.h>
#endif

#if !defined(__riscv_zfh) || !defined(__riscv_zvfh)
#error "riscv zfh extension not enabled, GGML_RV_ZFH and GGML_RV_ZVFH must be defined to 1"
#endif

#if !defined(__riscv_zba)
#error "riscv zba extension not enabled, GGML_RV_ZBA must be defined to 1"
#endif

#if defined(RISCV64_SPACEMIT_IME1) || defined(RISCV64_SPACEMIT_IME2)
#else
#error "RISCV64_SPACEMIT_IME1 or RISCV64_SPACEMIT_IME2 not defined"
#endif

#else

#error "riscv not enabled in this build"

#endif

#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Woverlength-strings"
#pragma GCC diagnostic ignored "-Wcast-qual"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

// clang-format on

extern "C" {
extern void ggml_threadpool_chunk_set(struct ggml_threadpool * tp, int value);
extern int  ggml_threadpool_chunk_add(struct ggml_threadpool * tp, int value);
}

namespace ggml::cpu::riscv64_spacemit {

struct TLSContext {
    int       cpu_id{ -1 };
    cpu_set_t cpuset;
    void *    tcm_buffer{ nullptr };
    size_t    tcm_buffer_size{ 0 };
};

thread_local TLSContext tls_context;

extern "C" int ggml_threadpool_graph_id(struct ggml_threadpool * tp);

// ---- cross-node A quantisation cache (env GGML_SPACEMIT_A_DEDUP, default on) ----
//
// Several MUL_MAT nodes consume the *same* activation with the same (M, K, block length):
// attn_post_norm -> ffn_up + ffn_gate, attn_norm -> wqkv + wqkv_gate, attn_norm -> wq/wk/wv,
// ced_attn_norm -> up/gate, and l_out-15 -> the four CED k/v projections.  Today each of them
// re-quantises the whole activation into its own per-node workspace.  Measured redundancy in
// the production model: 149 quantise passes per ubatch over only 94 distinct activations;
// 64% of the fp32 bytes the quantisers read are re-reads of an activation already quantised.
//
// Correctness rests on four things:
//   * the key is (graph id, tensor name, M, K, block length).  The graph id comes from
//     ggml_threadpool::n_graph, which the pool bumps once per graph *before* the workers wake,
//     so every thread agrees on it; it is what invalidates the cache between ubatches.
//   * hit/miss is decided against the *committed* key, and thread 0 commits only after the
//     quantise barrier.  So during a node every thread sees the same key, and on a miss every
//     thread fills its own row slice -- the fill is collective.
//   * the buffer is allocated once via std::call_once and never reallocated; a call whose
//     workspace does not fit simply bypasses the cache.
//   * single-threaded pools do not advance n_graph, so the cache is disabled when nth == 1.
// Direct-mapped: 8 slots, each big enough for the largest *duplicated* activation
// (M=512, K=2048 -> 1.03 MiB).  K=6144 activations need 3.2 MiB and have no duplicate
// consumer, so they simply bypass the cache.
static const int    SPX_A_DEDUP_SLOTS      = 8;
static const size_t SPX_A_DEDUP_SLOT_BYTES = (size_t) 2 << 20;  // 16 MiB total, allocated once

struct ADedupCache {
    bool      on  = false;
    uint8_t * buf = nullptr;

    int64_t gen[SPX_A_DEDUP_SLOTS];
    int64_t m[SPX_A_DEDUP_SLOTS];
    int64_t k[SPX_A_DEDUP_SLOTS];
    int64_t blk[SPX_A_DEDUP_SLOTS];
    char    name[SPX_A_DEDUP_SLOTS][48];

    ADedupCache() {
        const char * e = getenv("GGML_SPACEMIT_A_DEDUP");
        on             = (e == nullptr || e[0] != '0');
        for (int i = 0; i < SPX_A_DEDUP_SLOTS; i++) {
            gen[i] = -1; m[i] = k[i] = blk[i] = 0; name[i][0] = 0;
        }
    }
    bool usable(int nth) const { return on && nth > 1; }

    // every thread derives the same slot from the key, so the fill is collective and the
    // commit target is unambiguous
    static int slot_of(const char * nm, int64_t mm, int64_t kk, int64_t bb) {
        uint64_t h = 1469598103934665603ULL;
        for (const char * p = nm; p != nullptr && *p != 0; ++p) {
            h ^= (uint64_t) (uint8_t) *p;
            h *= 1099511628211ULL;
        }
        h ^= (uint64_t) mm * 1000003ULL;
        h *= 1099511628211ULL;
        h ^= (uint64_t) kk * 1000033ULL;
        h *= 1099511628211ULL;
        h ^= (uint64_t) bb;
        return (int) (h % (uint64_t) SPX_A_DEDUP_SLOTS);
    }
    bool ensure(size_t need) {
        if (need > SPX_A_DEDUP_SLOT_BYTES) {
            return false;
        }
        static std::once_flag once;
        std::call_once(once, [this] {
            buf = (uint8_t *) malloc(SPX_A_DEDUP_SLOT_BYTES * (size_t) SPX_A_DEDUP_SLOTS);
        });
        return buf != nullptr;
    }
    uint8_t * slot(int s) const { return buf + (size_t) s * SPX_A_DEDUP_SLOT_BYTES; }
    bool match(int s, int64_t g, const char * nm, int64_t mm, int64_t kk, int64_t bb) const {
        return gen[s] == g && m[s] == mm && k[s] == kk && blk[s] == bb && nm != nullptr &&
               strcmp(name[s], nm) == 0;
    }
    void commit(int s, int64_t g, const char * nm, int64_t mm, int64_t kk, int64_t bb) {
        gen[s] = g;
        m[s]   = mm;
        k[s]   = kk;
        blk[s] = bb;
        snprintf(name[s], sizeof(name[s]), "%s", nm != nullptr ? nm : "");
    }
};
static ADedupCache g_adedup;

// ---- A-offset fold (env GGML_SPACEMIT_A_FOLD=1) ----------------------------
// The Q4_0_K128 M4 inner loop carries 8 `vadd.vi -8` per iteration to undo the B nibble
// zero point.  Since
//     sum_k a*(q-8) = sum_k a*q - 8*sum_k a
// and `sum_k a` only depends on the (4-row, K-block) A slice, the correction can be applied
// once per K block instead of once per iteration.  quantize_a_4row_i8_with_sum() emits it as
// two int32 row-broadcast patterns (64 B) appended to each 592 B A block record, so the A
// stride grows by 16 B per row per K block.  Integer arithmetic throughout => bit-identical.
static bool spx_a_fold_enabled() {
    static const bool on = [] {
        const char * e = getenv("GGML_SPACEMIT_A_FOLD");
        return e == nullptr || e[0] != '0';  // default ON; GGML_SPACEMIT_A_FOLD=0 restores the
                                             // per-iteration B zero-point shift
    }();
    return on;
}

static constexpr size_t spx_a_fold_extra_per_blk() {
    // two int32 row-broadcast patterns (Sa, Sb) per 4-row x K-block group = 64 B,
    // i.e. 16 B per row per K block on top of q8_blk_size(blk_len)
    return 4 * sizeof(int32_t);
}
// hits / misses, printed when GGML_SPACEMIT_A_TRAFFIC=1 is set
struct ADedupStats {
    int64_t hit = 0, miss = 0, bypass = 0;
    ~ADedupStats() {
        const char * e = getenv("GGML_SPACEMIT_A_TRAFFIC");
        if (e != nullptr && e[0] == '1' && (hit || miss)) {
            fprintf(stderr, "[spxdedup] quantise passes: hit=%ld miss=%ld bypass=%ld (hit rate %.1f%%)\n",
                    (long) hit, (long) miss, (long) bypass,
                    100.0 * (double) hit / (double) (hit + miss));
        }
    }
};
static ADedupStats g_adedup_stats;

template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS> constexpr size_t get_repacked_block_type_size() {
    if constexpr (std::is_same_v<BLOC_TYPE, block_q6_K> || std::is_same_v<BLOC_TYPE, block_q8_0>) {
        return sizeof(block_q8_0);
    } else if constexpr (std::is_same_v<BLOC_TYPE, block_q4_0>) {
        return sizeof(block_q4_0) * INTER_SIZE / QK4_0;
    } else if constexpr (std::is_same_v<BLOC_TYPE, block_q4_0_k64>) {
        return sizeof(block_q4_0_k64) * INTER_SIZE / QK4_0_K64;
    } else if constexpr (std::is_same_v<BLOC_TYPE, block_q4_0_k128>) {
        return sizeof(block_q4_0_k128) * INTER_SIZE / QK4_0_K128;
    } else if constexpr (std::is_same_v<BLOC_TYPE, block_q4_1> || std::is_same_v<BLOC_TYPE, block_q4_K>) {
        return (sizeof(block_q4_0) + sizeof(uint8_t)) * INTER_SIZE / QK4_1;
    } else if constexpr (std::is_same_v<BLOC_TYPE, block_q2_K>) {
        return sizeof(spacemit_kernels::nrow_block_q2_k<1>);
    } else if constexpr (std::is_same_v<BLOC_TYPE, block_q3_K>) {
        return sizeof(spacemit_kernels::nrow_block_q3_k<1>);
    } else if constexpr (std::is_same_v<BLOC_TYPE, block_mxfp4>) {
        return sizeof(spacemit_kernels::nrow_block_mxfp4<1>);
    } else if constexpr (std::is_same_v<BLOC_TYPE, block_q5_1> || std::is_same_v<BLOC_TYPE, block_q5_K>) {
        return sizeof(spacemit_kernels::nrow_block_q5_1<1>);
    } else if constexpr (std::is_same_v<BLOC_TYPE, block_q5_0>) {
        return sizeof(spacemit_kernels::nrow_block_q5_0<1>);
    } else {
        assert(false);
        return 0;
    }
}

template <typename BLOC_TYPE> constexpr bool block_type_has_zp() {
    if constexpr (std::is_same_v<BLOC_TYPE, block_q6_K> || std::is_same_v<BLOC_TYPE, block_q8_0> ||
                  std::is_same_v<BLOC_TYPE, block_q3_K> || std::is_same_v<BLOC_TYPE, block_q4_0> ||
                  std::is_same_v<BLOC_TYPE, block_mxfp4> || std::is_same_v<BLOC_TYPE, block_q5_0>) {
        return false;
    } else if constexpr (std::is_same_v<BLOC_TYPE, block_q4_1> || std::is_same_v<BLOC_TYPE, block_q4_K> ||
                         std::is_same_v<BLOC_TYPE, block_q2_K> || std::is_same_v<BLOC_TYPE, block_q5_1> ||
                         std::is_same_v<BLOC_TYPE, block_q5_K>) {
        return true;
    } else {
        assert(false);
        return false;
    }
}

class tensor_traits_base : public ggml::cpu::tensor_traits {
  public:
    virtual int repack(ggml_tensor * t, const void * data, size_t data_size) = 0;
};

template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS> class tensor_traits : public tensor_traits_base {
    bool work_size(int /* n_threads */, const ggml_tensor * op, size_t & size) override {
        switch (op->op) {
            case GGML_OP_MUL_MAT:
                {
                    int64_t src1_nelements = ggml_nelements(op->src[1]);

                    if constexpr (std::is_same_v<BLOC_TYPE, block_q2_K> || std::is_same_v<BLOC_TYPE, block_q3_K>) {
                        size =
                            spacemit_kernels::div_round_up(src1_nelements, QK_K) * spacemit_kernels::q8k_blk_size(QK_K);
                    } else if constexpr (INTER_SIZE == QK4_0 || INTER_SIZE == QK4_0_K64 || INTER_SIZE == QK4_0_K128) {
                        size = spacemit_kernels::div_round_up(src1_nelements, INTER_SIZE) *
                               spacemit_kernels::q8_blk_size(INTER_SIZE, true);
                        if constexpr (INTER_SIZE == QK4_0_K128) {
                            // [A-offset fold] the folded A record is q8_blk_size(128) + 16 B per row
                            if (spx_a_fold_enabled()) {
                                size = spacemit_kernels::div_round_up(src1_nelements, INTER_SIZE) *
                                       (spacemit_kernels::q8_blk_size(INTER_SIZE) + spx_a_fold_extra_per_blk());
                            }
                        }
                    } else if constexpr (INTER_SIZE == 256) {
                        size = spacemit_kernels::div_round_up(src1_nelements, 256) *
                               spacemit_kernels::q8_hp_blk_size(256, true, true);
                    } else {
                        GGML_ABORT("unsupported block type");
                    }

                    size = GGML_PAD(size, sizeof(int64_t));

                    return true;
                }
            case GGML_OP_MUL_MAT_ID:
                {
                    int64_t src1_nelements = ggml_nelements(op->src[1]);

                    if constexpr (std::is_same_v<BLOC_TYPE, block_q2_K> || std::is_same_v<BLOC_TYPE, block_q3_K>) {
                        size =
                            spacemit_kernels::div_round_up(src1_nelements, QK_K) * spacemit_kernels::q8k_blk_size(QK_K);
                    } else if constexpr (INTER_SIZE == QK4_0 || INTER_SIZE == QK4_0_K64 || INTER_SIZE == QK4_0_K128) {
                        size = spacemit_kernels::div_round_up(src1_nelements, INTER_SIZE) *
                               spacemit_kernels::q8_blk_size(INTER_SIZE, true);
                    } else if constexpr (INTER_SIZE == 256) {
                        size = spacemit_kernels::div_round_up(src1_nelements, 256) *
                               spacemit_kernels::q8_hp_blk_size(256, true, true);
                    } else {
                        GGML_ABORT("unsupported block type");
                    }

                    size = GGML_PAD(size, sizeof(int64_t));

                    const int64_t ne02 = op->src[0]->ne[2];  // n_as, n_expert
                    const int64_t ne12 = op->src[1]->ne[2];  // n_tokens

                    const size_t sizeof_mmid_row_mapping = sizeof(int64_t);
                    size += sizeof_mmid_row_mapping * ne02 * (ne12 + 1) + (ne02 + 1) * sizeof(int64_t);

                    size = GGML_PAD(size, sizeof(int64_t));

                    return true;
                }
            default:
                // GGML_ABORT("fatal error");
                break;
        }
        return false;
    }

    bool compute_forward(ggml_compute_params * params, ggml_tensor * op) override {
        switch (op->op) {
            case GGML_OP_MUL_MAT:
                switch (op->src[0]->type) {
                    case GGML_TYPE_Q2_K:
                    case GGML_TYPE_Q3_K:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q4_K:
                    case GGML_TYPE_Q6_K:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q5_K:
                    case GGML_TYPE_Q4_0_K64:
                    case GGML_TYPE_Q4_0_K128:
                        //case GGML_TYPE_MXFP4:
                        forward_mul_mat(params, op);
                        return true;
                    default:
                        // GGML_ABORT("fatal error: unsupported type for src0 in MUL_MAT");
                        return false;
                }
                break;
            case GGML_OP_MUL_MAT_ID:
                switch (op->src[0]->type) {
                    case GGML_TYPE_Q2_K:
                    case GGML_TYPE_Q3_K:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q4_K:
                    case GGML_TYPE_Q6_K:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q5_K:
                    case GGML_TYPE_Q4_0_K64:
                    case GGML_TYPE_Q4_0_K128:
                        //case GGML_TYPE_MXFP4:
                        forward_mul_mat_id(params, op);
                        return true;
                    default:
                        // GGML_ABORT("fatal error: unsupported type for src0 in MUL_MAT_ID");
                        return false;
                }
                break;
            default:
                // GGML_ABORT("fatal error");
                break;
        }
        return false;
    }

    void forward_mul_mat(ggml_compute_params * params, ggml_tensor * op) {
        constexpr size_t a_blk_len = INTER_SIZE;
        constexpr size_t b_blk_len = INTER_SIZE;

        const ggml_tensor * src0 = op->src[0];
        const ggml_tensor * src1 = op->src[1];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_BINARY_OP_LOCALS

        int ith = params->ith;
        int nth = params->nth;

        [[maybe_unused]] const enum ggml_type type = src0->type;

        void *        w_data  = (void *) src0->data;
        const float * feature = (const float *) src1->data;
        float *       output  = (float *) dst->data;

        const int64_t gemm_m = ne11 * ne12 * ne13;
        const int64_t gemm_k = ne10;
        const int64_t gemm_n = ne01;

        spacemit_kernels::quantize_a_row_def       quantize_a_row_i8;
        spacemit_kernels::quantize_a_row_def       quantize_a_4row_i8;
        spacemit_kernels::gemm_kernel_quantize_def gemm_kernel;
        bool                                       set_kernel_impl = false;

        int64_t block_stride_a = spacemit_kernels::q8_blk_size(a_blk_len);

#if defined(RISCV64_SPACEMIT_IME2)
        if (!set_kernel_impl && (global_spine_env_info.use_ime2)) {
            quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8;
            quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8;
            block_stride_a     = spacemit_kernels::q8_blk_size(a_blk_len, true);

            if constexpr (std::is_same_v<BLOC_TYPE, block_q6_K> || std::is_same_v<BLOC_TYPE, block_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i8;
                set_kernel_impl = true;
            } else if constexpr (std::is_same_v<BLOC_TYPE, block_q4_0> || std::is_same_v<BLOC_TYPE, block_q4_1> ||
                                 std::is_same_v<BLOC_TYPE, block_q4_K>) {
                if constexpr (INTER_SIZE == 256) {
                    gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i4_hp;
                    quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8_hp;
                    quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8_hp;
                    block_stride_a     = spacemit_kernels::q8_hp_blk_size(a_blk_len, true, true);
                    set_kernel_impl    = true;
                } else {
                    gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i4;
                    quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8;
                    quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8;
                    block_stride_a     = spacemit_kernels::q8_blk_size(a_blk_len, true);
                    set_kernel_impl    = true;
                }
            } else if constexpr (std::is_same_v<BLOC_TYPE, block_q2_K>) {
                quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8k;
                quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8k;
                block_stride_a     = spacemit_kernels::q8k_blk_size(a_blk_len);

                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i2k;
                set_kernel_impl = true;
            } else if constexpr (std::is_same_v<BLOC_TYPE, block_q3_K>) {
                quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8k;
                quantize_a_4row_i8 = spacemit_kernels::rvv::quantize_a_4row_i8k;
                block_stride_a     = spacemit_kernels::q8k_blk_size(a_blk_len);

                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i3k;
                set_kernel_impl = true;
            } else if constexpr (std::is_same_v<BLOC_TYPE, block_mxfp4>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8mxfp4;
                set_kernel_impl = true;
            } else if constexpr (std::is_same_v<BLOC_TYPE, block_q5_1> || std::is_same_v<BLOC_TYPE, block_q5_K> ||
                                 std::is_same_v<BLOC_TYPE, block_q5_0>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i5;
                set_kernel_impl = true;
            }
        }
#endif

#if defined(RISCV64_SPACEMIT_IME1)
        if (!set_kernel_impl && (global_spine_env_info.use_ime1)) {
            quantize_a_row_i8  = spacemit_kernels::ime1::quantize_a_row_i8;
            quantize_a_4row_i8 = spacemit_kernels::ime1::quantize_a_4row_i8;

            if constexpr (std::is_same_v<BLOC_TYPE, block_q4_0> || std::is_same_v<BLOC_TYPE, block_q4_1> ||
                          std::is_same_v<BLOC_TYPE, block_q4_K> || std::is_same_v<BLOC_TYPE, block_q4_0_k64> ||
                          std::is_same_v<BLOC_TYPE, block_q4_0_k128>) {
                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i4;
                set_kernel_impl = true;
            } else if constexpr (std::is_same_v<BLOC_TYPE, block_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i8;
                set_kernel_impl = true;
            }
        }
#endif
        if (!set_kernel_impl) {
            GGML_ABORT("no kernel implementation found for the block type");
        }

        // [A-offset fold] switch the Q4_0_K128 M4 path to the folded zero-point kernel and the
        // matching A layout.  Requires gemm_m % 4 == 0 because the tail rows would otherwise be
        // quantised by the single-row quantiser, which does not emit the block sums.
        if (spx_a_fold_enabled() && std::is_same_v<BLOC_TYPE, block_q4_0_k128> && gemm_m > 1 &&
            (gemm_m % 4) == 0) {
            block_stride_a     = spacemit_kernels::q8_blk_size(a_blk_len) + spx_a_fold_extra_per_blk();
            quantize_a_4row_i8 = spacemit_kernels::ime1::quantize_a_4row_i8_with_sum;
            gemm_kernel        = spacemit_kernels::ime1::gemm_kernel_i8i4_fold;
        }

        const int64_t a_k_blks = spacemit_kernels::div_round_up(gemm_k, a_blk_len);
        const int64_t b_k_blks = spacemit_kernels::div_round_up(gemm_k, b_blk_len);

        const int64_t row_stride_a        = a_k_blks * block_stride_a;
        const int64_t gemm_workspace_size = GGML_PAD(gemm_m * row_stride_a, alignof(int64_t));

        if (ith == 0 && params->wsize < gemm_workspace_size) {
            GGML_ABORT("wsize less than gemm_workspace_size");
        }

        uintptr_t ws_ptr = reinterpret_cast<uintptr_t>(params->wdata);

        void *        tcm_buffer      = ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer;
        const int64_t tcm_buffer_size = ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer_size;

        uint8_t * quant_a = reinterpret_cast<uint8_t *>(ws_ptr);

        constexpr int64_t row_align = 4;
        const int64_t     row_blks  = spacemit_kernels::div_round_up(gemm_m, row_align);

        const int64_t row_stride_b      = b_k_blks * get_repacked_block_type_size<BLOC_TYPE, INTER_SIZE, NB_COLS>();
        const int64_t per_mb_rows_wsize = row_align * row_stride_a;
        const int64_t per_nb_cols_wsize = NB_COLS * row_stride_b;

        const int64_t barrier_idx = static_cast<int64_t>(ith / 2);

        GGML_ASSERT(global_spine_env_info.init_barrier != nullptr);
        GGML_ASSERT(barrier_idx < spine_init_barrier_count);
        spine_barrier_t * cur_barrier = &global_spine_env_info.init_barrier[barrier_idx];


        // ---- prefill phase profiler: GGML_SPACEMIT_MM_PHASES=1 -------------
        struct MmPhaseStats {
            struct Ent { int64_t m, n, k; char path; int64_t cnt, tq, tb, tg, tbc, bbc, aread; };
            Ent     e[512];
            int     n = 0;
            bool    on = false;
            MmPhaseStats() {
                const char * s = getenv("GGML_SPACEMIT_MM_PHASES");
                on = (s != NULL && s[0] == '1');
            }
            void add(int64_t m, int64_t n_, int64_t k, char path, int64_t tq, int64_t tb, int64_t tg,
                     int64_t tbc, int64_t bbc, int64_t aread) {
                if (!on) {
                    return;
                }
                for (int i = 0; i < n; i++) {
                    if (e[i].m == m && e[i].n == n_ && e[i].k == k && e[i].path == path) {
                        e[i].cnt++; e[i].tq += tq; e[i].tb += tb; e[i].tg += tg;
                        e[i].tbc += tbc; e[i].bbc += bbc; e[i].aread += aread;
                        return;
                    }
                }
                if (n < (int) (sizeof(e) / sizeof(e[0]))) {
                    e[n].m = m; e[n].n = n_; e[n].k = k; e[n].path = path;
                    e[n].cnt = 1; e[n].tq = tq; e[n].tb = tb; e[n].tg = tg;
                    e[n].tbc = tbc; e[n].bbc = bbc; e[n].aread = aread;
                    n++;
                }
            }
            ~MmPhaseStats() {
                if (!on || n == 0) {
                    return;
                }
                int64_t T = 0;
                for (int i = 0; i < n; i++) {
                    T += e[i].tq + e[i].tb + e[i].tg;
                }
                fprintf(stderr, "[spxmm] ===== pid=%d spacemit MUL_MAT phases (thread0, wall) total %ld us =====\n",
                        (int) getpid(), (long) T);
                fprintf(stderr, "[spxmm] %-6s %-7s %-7s %-5s %-6s %-10s %-8s %-12s %-10s %-10s %-10s\n",
                        "M", "N", "K", "path", "n", "quant_us", "bar_us", "gemm_us", "bcopy_us", "bcopy_B", "a_read_MB");
                for (int i = 0; i < n; i++) {
                    fprintf(stderr, "[spxmm] %-6ld %-7ld %-7ld %-5c %-6ld %-10ld %-8ld %-12ld %-10ld %-10ld %-10.2f\n",
                            (long) e[i].m, (long) e[i].n, (long) e[i].k, e[i].path, (long) e[i].cnt,
                            (long) e[i].tq, (long) e[i].tb, (long) e[i].tg,
                            (long) e[i].tbc, (long) e[i].bbc, (double) e[i].aread / 1048576.0);
                }
                fprintf(stderr, "[spxmm] sums: quant=%ld bar=%ld gemm=%ld\n",
                        (long) [&]{ int64_t s = 0; for (int i = 0; i < n; i++) s += e[i].tq; return s; }(),
                        (long) [&]{ int64_t s = 0; for (int i = 0; i < n; i++) s += e[i].tb; return s; }(),
                        (long) [&]{ int64_t s = 0; for (int i = 0; i < n; i++) s += e[i].tg; return s; }());
            }
        };
        static MmPhaseStats g_mmph;
        const bool    spx_mm  = g_mmph.on;
        const int64_t spx_tq0 = spx_mm ? ggml_time_us() : 0;
        int64_t       spx_bc_us = 0, spx_bc_bytes = 0, spx_aread = 0;

        // ---- optional cross-node A quantisation cache ----
        bool      a_dedup_use   = false;
        bool      a_dedup_hit   = false;
        int64_t   a_dedup_gen   = 0;
        int       a_dedup_slot  = -1;
        if (g_adedup.usable(nth) && gemm_m > 1 && src1->name[0] != 0) {
            const size_t need = (size_t) gemm_m * (size_t) row_stride_a;
            if (g_adedup.ensure(need)) {
                a_dedup_slot = ADedupCache::slot_of(src1->name, gemm_m, gemm_k, (int64_t) a_blk_len);
                a_dedup_gen  = ggml_threadpool_graph_id(params->threadpool);
                a_dedup_use  = true;
                quant_a      = g_adedup.slot(a_dedup_slot);
                a_dedup_hit  = g_adedup.match(a_dedup_slot, a_dedup_gen, src1->name, gemm_m, gemm_k,
                                              (int64_t) a_blk_len);
            } else if (ith == 0) {
                g_adedup_stats.bypass++;
            }
        }
        if (a_dedup_use && ith == 0) {
            if (a_dedup_hit) { g_adedup_stats.hit++; } else { g_adedup_stats.miss++; }
        }
        if (!a_dedup_hit) {
        if (gemm_m == 1) {
            int task_per_thread = spacemit_kernels::div_round_up(a_k_blks, nth);
            int a_blk_start     = ith * task_per_thread;
            int a_blk_end       = std::min(a_blk_start + task_per_thread, (int) a_k_blks);
            if (a_blk_start < a_blk_end) {
                quantize_a_row_i8(a_blk_len, feature + a_blk_start * a_blk_len, (a_blk_end - a_blk_start) * a_blk_len,
                                  quant_a + a_blk_start * block_stride_a);
            }
        } else {
            int task_per_thread = spacemit_kernels::div_round_up(row_blks, nth);
            int m_row_blk_start = ith * task_per_thread;
            int m_row_blk_end   = std::min(m_row_blk_start + task_per_thread, (int) row_blks);
            for (int m_row_blk = m_row_blk_start; m_row_blk < m_row_blk_end; m_row_blk++) {
                int m_idx             = m_row_blk * row_align;
                int rows_tobe_handled = (gemm_m - m_idx) > row_align ? row_align : (gemm_m - m_idx);

                if (rows_tobe_handled == row_align && quantize_a_4row_i8 != nullptr) {
                    const float * a_row_ptr       = feature + m_idx * gemm_k;
                    auto *        quant_a_row_ptr = quant_a + m_idx * row_stride_a;
                    quantize_a_4row_i8(a_blk_len, a_row_ptr, gemm_k, quant_a_row_ptr);
                } else {
                    while (rows_tobe_handled) {
                        const float * a_row_ptr       = feature + m_idx * gemm_k;
                        auto *        quant_a_row_ptr = quant_a + m_idx * row_stride_a;
                        quantize_a_row_i8(a_blk_len, a_row_ptr, gemm_k, quant_a_row_ptr);
                        rows_tobe_handled -= 1;
                        m_idx += 1;
                    }
                }
            }
        }
        }  // !a_dedup_hit

        const int64_t spx_tq1 = spx_mm ? ggml_time_us() : 0;

        ggml_barrier(params->threadpool);

        if (a_dedup_use && ith == 0) {
            // every thread has finished writing its row slice (barrier above), so the buffer is
            // complete and the key can be published for the next consumer of this activation
            g_adedup.commit(a_dedup_slot, a_dedup_gen, src1->name, gemm_m, gemm_k, (int64_t) a_blk_len);
        }

        const int64_t spx_tb1 = spx_mm ? ggml_time_us() : 0;

        const int64_t gemm_m_stride     = gemm_n / gemm_m > 64 ? gemm_m : 16;
        const int64_t gemm_m_blocked    = spacemit_kernels::div_round_up(gemm_m, gemm_m_stride);
        const int64_t max_gemm_n_stride = spacemit_kernels::div_round_up(gemm_n * gemm_m_blocked, nth);

        int64_t gemm_n_stride = gemm_n;
        if (max_gemm_n_stride < gemm_n) {
            gemm_n_stride =
                std::min(gemm_n_stride, spacemit_kernels::div_round_up(max_gemm_n_stride, NB_COLS) * NB_COLS);
        }

        // path selection: the TCM paths (A/B) re-read the weights once per m-block, which
        // only pays off for the single-row decode case; prefill (m>1) is faster in Path C
        static int mm_path = -2;
        if (mm_path == -2) {
            const char * e = getenv("GGML_SPACEMIT_MM_PATH");
            // default = Path F (weight-stationary, one B n-tile staged in TCM and every m row
            // streamed through it) for prefill: measured +7.2% pp512 over Path G on
            // Qwen3.5-2B/Q4_0_K64 (3/3 interleaved rounds).  When its slab precondition fails the
            // chain falls through to the generic path.  Set GGML_SPACEMIT_MM_PATH to override.
            mm_path = (e != NULL) ? (int) e[0] : 'f';
        }
        bool prefer_tcm = (gemm_m == 1);
        if (mm_path == 'b' || mm_path == 'B') {
            prefer_tcm = true;
        } else if (mm_path == 'c' || mm_path == 'C') {
            prefer_tcm = false;
        }

        // Path B slab size in columns (env GGML_SPACEMIT_TCM_SLAB, in units of NB_COLS): a larger
        // slab streams longer from DDR per barrier, bounded by the per-thread TCM block size
        int64_t slab_cols;
        {
            static int64_t slab_env = -1;
            if (slab_env < 0) {
                const char * e = getenv("GGML_SPACEMIT_TCM_SLAB");
                int          mult = (e != NULL) ? atoi(e) : 1;
                if (mult < 1) {
                    mult = 1;
                }
                slab_env = NB_COLS * mult;
            }
            slab_cols = slab_env;
        }

        // Path F n-tile size: as many output columns as fit in TCM alongside nothing else.
        int64_t ws_slab = (static_cast<int64_t>(tcm_buffer_size) / row_stride_b) / NB_COLS * NB_COLS;
        if (ws_slab < NB_COLS) {
            ws_slab = 0; // does not fit -> Path F disabled
        } else if (ws_slab > gemm_n) {
            ws_slab = gemm_n;
        }

        // Path D (decode only): stream the repacked weights straight from DDR with every thread
        // owning a contiguous run of 16-column groups. Path B's even/odd ping-pong keeps only half
        // the threads reading DDR at a time, which caps the achievable bandwidth; here all threads
        // stream, and the M=1 kernel prefetches the next K-block to hide the DDR latency.
        const bool ime1_int4 = std::is_same_v<BLOC_TYPE, block_q4_0> ||
                               std::is_same_v<BLOC_TYPE, block_q4_1> ||
                               std::is_same_v<BLOC_TYPE, block_q4_K>;
        const bool vlen256 = (__riscv_vlenb() * 8 == 256);

        if (gemm_m == 1 && NB_COLS == 16 && ime1_int4 && vlen256 &&
            (mm_path == 'd' || mm_path == 'D')) {
            // ceil, not floor: NB_COLS need not divide gemm_n (e.g. 2048 % 20 == 8); the last
            // group is narrower and the M=1 kernel's tail path writes only the valid columns.
            const int64_t n_groups = spacemit_kernels::div_round_up(gemm_n, NB_COLS);
            const int64_t gpt      = spacemit_kernels::div_round_up(n_groups, nth);
            const int64_t g_start  = std::min(static_cast<int64_t>(ith) * gpt, n_groups);
            const int64_t g_end    = std::min(g_start + gpt, n_groups);

            for (int64_t g = g_start; g < g_end; ++g) {
                const int64_t n_off = g * NB_COLS;
                const int64_t n_cnt = std::min<int64_t>(NB_COLS, gemm_n - n_off);
                uint8_t *     b_col = reinterpret_cast<uint8_t *>(w_data) + n_off * row_stride_b;
                uint8_t *     b_zp  = block_type_has_zp<BLOC_TYPE>() ? b_col : nullptr;

                gemm_kernel(b_blk_len, quant_a, b_col, b_zp, output + n_off,
                            /*count_m=*/1, NB_COLS, b_k_blks, gemm_n);
            }
        } else if ((mm_path == 'g' || mm_path == 'G') && gemm_m > 1 && gemm_n_stride == gemm_n &&
                   tcm_buffer != nullptr &&
                   spacemit_kernels::div_round_up(spacemit_kernels::div_round_up(gemm_m, row_align), nth) *
                           per_mb_rows_wsize <=
                       tcm_buffer_size) {
            // Path G (N-major, ported from the other tree's 7732584 N-major commit):
            // stage ALL of this thread's M4 chunks into its TCM quota once, then walk the N tiles
            // outermost and the M chunks innermost, so one B tile serves every chunk back to back.
            const int64_t g_chunks = spacemit_kernels::div_round_up(spacemit_kernels::div_round_up(gemm_m, row_align), nth);
            {
                int64_t chunk = 0;
                for (int64_t m_start = ith * row_align; m_start < gemm_m; m_start += row_align * nth, chunk++) {
                    const int64_t m_row_real = std::min(gemm_m - m_start, row_align);
                    const int64_t spx_c1 = spx_mm ? ggml_time_us() : 0;
                    spacemit_kernels::rvv::memcpy1d(reinterpret_cast<uint8_t *>(tcm_buffer) + chunk * per_mb_rows_wsize,
                                                    quant_a + m_start * row_stride_a,
                                                    m_row_real * row_stride_a);
                    if (spx_mm) { spx_bc_us += ggml_time_us() - spx_c1; spx_bc_bytes += m_row_real * row_stride_a; }
                }
            }
            (void) g_chunks;
            for (int64_t ni = 0; ni < gemm_n; ni += NB_COLS) {
                const int64_t n_blk_real = std::min(gemm_n - ni, (int64_t) NB_COLS);
                uint8_t *     b_col      = reinterpret_cast<uint8_t *>(w_data) + ni * row_stride_b;
                uint8_t *     b_col_zp   = block_type_has_zp<BLOC_TYPE>() ? b_col : nullptr;
                int64_t       chunk2     = 0;
                for (int64_t m_start = ith * row_align; m_start < gemm_m; m_start += row_align * nth, chunk2++) {
                    const int64_t m_row_real  = std::min(gemm_m - m_start, row_align);
                    uint8_t *     a_row_ptr   = reinterpret_cast<uint8_t *>(tcm_buffer) + chunk2 * per_mb_rows_wsize;
                    float *       c_blk       = output + m_start * gemm_n + ni;
                    int32_t       rows_remaining = (int32_t) m_row_real;
                    while (rows_remaining > 0) {
                        auto rows_handled = gemm_kernel(b_blk_len, a_row_ptr, b_col, b_col_zp, c_blk, rows_remaining,
                                                        n_blk_real, b_k_blks, gemm_n);
                        c_blk += rows_handled * gemm_n;
                        a_row_ptr += rows_handled * row_stride_a;
                        rows_remaining -= rows_handled;
                    }
                }
            }
        } else if (prefer_tcm && gemm_n_stride == gemm_n && tcm_buffer != nullptr && per_mb_rows_wsize <= tcm_buffer_size) {
            for (int64_t m_start = ith * row_align; m_start < gemm_m; m_start += row_align * nth) {
                uint8_t * b_col    = reinterpret_cast<uint8_t *>(w_data);
                uint8_t * b_col_zp = block_type_has_zp<BLOC_TYPE>() ? b_col : nullptr;

                int64_t m_row_real = std::min(gemm_m - m_start, row_align);

                const int64_t spx_c2 = spx_mm ? ggml_time_us() : 0;
                spacemit_kernels::rvv::memcpy1d(tcm_buffer, quant_a + m_start * row_stride_a,
                                                m_row_real * row_stride_a);
                if (spx_mm) { spx_bc_us += ggml_time_us() - spx_c2; spx_bc_bytes += m_row_real * row_stride_a; }

                int64_t n_blk_real = 0;
                for (int64_t ni = 0; ni < gemm_n; ni += n_blk_real, b_col += n_blk_real * row_stride_b) {
                    n_blk_real = std::min(gemm_n - ni, (int64_t) NB_COLS);

                    uint8_t * a_row_ptr = (uint8_t *) tcm_buffer;
                    float *   c_blk     = output + m_start * gemm_n + ni;

                    int32_t rows_remaining = m_row_real;

                    while (rows_remaining > 0) {
                        auto rows_handled = gemm_kernel(b_blk_len, a_row_ptr, b_col, b_col_zp, c_blk, rows_remaining,
                                                        n_blk_real, b_k_blks, gemm_n);

                        c_blk += rows_handled * gemm_n;
                        a_row_ptr += rows_handled * row_stride_a;

                        rows_remaining -= rows_handled;
                    }
                }
            }
        } else if ((mm_path == 'e' || mm_path == 'E') && gemm_m == 1 && tcm_buffer != nullptr &&
                   (gemm_workspace_size + slab_cols * row_stride_b) <= tcm_buffer_size) {
            // Path E (experiment): every thread stages its own slab into its private TCM and then
            // computes it, with no pairwise spine barriers. All threads copy concurrently, so the DDR
            // stream is driven by nth threads instead of Path B's half; the trade-off is that a thread
            // no longer overlaps its copy with the partner's compute.
            uint8_t * a_row = (uint8_t *) tcm_buffer;
            uint8_t * b_buf = (uint8_t *) tcm_buffer + gemm_workspace_size;
            spacemit_kernels::rvv::memcpy1d(a_row, quant_a, gemm_workspace_size);
            uint8_t * b_zp = block_type_has_zp<BLOC_TYPE>() ? b_buf : nullptr;

            for (int64_t ni = static_cast<int64_t>(ith) * slab_cols; ni < gemm_n; ni += slab_cols * nth) {
                const int64_t nb_real = std::min(gemm_n - ni, slab_cols);
                spacemit_kernels::rvv::memcpy1d(b_buf, reinterpret_cast<uint8_t *>(w_data) + ni * row_stride_b,
                                                nb_real * row_stride_b);

                int64_t rows_remaining = gemm_m;
                float * c_blk          = output + ni;
                auto *  a_row_cur      = a_row;
                while (rows_remaining > 0) {
                    auto rows_handled = gemm_kernel(b_blk_len, a_row_cur, b_buf, b_zp, c_blk, rows_remaining,
                                                    nb_real, b_k_blks, gemm_n);
                    c_blk += rows_handled * gemm_n;
                    a_row_cur += rows_handled * row_stride_a;
                    rows_remaining -= rows_handled;
                }
            }
        } else if (prefer_tcm && tcm_buffer != nullptr && slab_cols * row_stride_b <= tcm_buffer_size) {
            uint8_t * a_row = quant_a;
            uint8_t * b_col = reinterpret_cast<uint8_t *>(tcm_buffer);
            if ((gemm_workspace_size + slab_cols * row_stride_b) <= tcm_buffer_size) {
                a_row = (uint8_t *) tcm_buffer;
                b_col = reinterpret_cast<uint8_t *>(tcm_buffer) + gemm_workspace_size;
            }
            uint8_t * b_col_zp = block_type_has_zp<BLOC_TYPE>() ? b_col : nullptr;

            int64_t ni      = ith * slab_cols;
            int64_t nb_real = std::min(gemm_n - ni, slab_cols);

            if (ith % 2 == 0 && nb_real > 0) {
                spacemit_kernels::rvv::memcpy1d(b_col, reinterpret_cast<uint8_t *>(w_data) + ni * row_stride_b,
                                                nb_real * row_stride_b);
                if (a_row != quant_a) {
                    spacemit_kernels::rvv::memcpy1d(a_row, quant_a, gemm_workspace_size);
                }
            }

            spine_barrier_wait(cur_barrier);

            if (ith % 2 != 0 && nb_real > 0) {
                if (a_row != quant_a) {
                    spacemit_kernels::rvv::memcpy1d(a_row, quant_a, gemm_workspace_size);
                }
                spacemit_kernels::rvv::memcpy1d(b_col, reinterpret_cast<uint8_t *>(w_data) + ni * row_stride_b,
                                                nb_real * row_stride_b);
            }

            for (; ni < gemm_n; ni += slab_cols * nth) {
                int64_t rows_remaining = gemm_m;
                float * c_blk          = output + ni;
                auto *  a_row_cur      = a_row;

                if (ith % 2 != 0) {
                    spine_barrier_wait(cur_barrier);
                }

                while (rows_remaining > 0) {
                    auto rows_handled = gemm_kernel(b_blk_len, a_row_cur, b_col, b_col_zp, c_blk, rows_remaining,
                                                    nb_real, b_k_blks, gemm_n);

                    c_blk += rows_handled * gemm_n;
                    a_row_cur += rows_handled * row_stride_a;

                    rows_remaining -= rows_handled;
                }

                if (ith % 2 == 0) {
                    spine_barrier_wait(cur_barrier);
                }

                const int64_t next_ni = ni + slab_cols * nth;
                if (next_ni < gemm_n) {
                    nb_real = std::min(gemm_n - next_ni, slab_cols);
                    spacemit_kernels::rvv::memcpy1d(b_col, reinterpret_cast<uint8_t *>(w_data) + next_ni * row_stride_b,
                                                    nb_real * row_stride_b);
                }
            }
        } else if ((mm_path == 'f' || mm_path == 'F') && gemm_m > 1 && ws_slab > 0 && tcm_buffer != nullptr) {
            // Path F: weight-stationary prefill. The M=4 kernel consumes only 4 rows per pass, so
            // Path C re-reads each weight tile once per 4 rows (and once per 16-row m-tile). Here one
            // n-tile of B is staged into TCM once and every m row is streamed through it, so the
            // weights are read from DDR exactly once per (thread, n-tile).
            uint8_t * b_tcm = reinterpret_cast<uint8_t *>(tcm_buffer);
            uint8_t * b_zp  = block_type_has_zp<BLOC_TYPE>() ? b_tcm : nullptr;

            for (int64_t n0 = static_cast<int64_t>(ith) * ws_slab; n0 < gemm_n; n0 += ws_slab * nth) {
                const int64_t nb_real = std::min(gemm_n - n0, ws_slab);

                const int64_t spx_c0 = spx_mm ? ggml_time_us() : 0;
                spacemit_kernels::rvv::memcpy1d(b_tcm, reinterpret_cast<uint8_t *>(w_data) + n0 * row_stride_b,
                                                nb_real * row_stride_b);
                if (spx_mm) { spx_bc_us += ggml_time_us() - spx_c0; spx_bc_bytes += nb_real * row_stride_b; }

                for (int64_t m0 = 0; m0 < gemm_m; ) {
                    const int64_t rows = std::min(gemm_m - m0, static_cast<int64_t>(row_align));
                    if (spx_mm) { spx_aread += rows * row_stride_a; }
                    auto rows_handled  = gemm_kernel(b_blk_len, quant_a + m0 * row_stride_a, b_tcm, b_zp,
                                                     output + m0 * gemm_n + n0, rows, nb_real, b_k_blks, gemm_n);
                    m0 += rows_handled;
                }
            }
        } else {
            const int64_t task_count_m = spacemit_kernels::div_round_up(gemm_m, gemm_m_stride);
            const int64_t task_count_n = spacemit_kernels::div_round_up(gemm_n, gemm_n_stride);

            int64_t task_count      = task_count_m * task_count_n;
            int64_t task_per_thread = (task_count + nth - 1) / nth;
            int64_t start           = ith * task_per_thread;
            int64_t end             = std::min((ith + 1) * task_per_thread, task_count);
            for (int64_t compute_idx = start; compute_idx < end; compute_idx++) {
                const auto tid_n = compute_idx / task_count_m;
                const auto tid_m = compute_idx % task_count_m;

                const int64_t m_start = tid_m * gemm_m_stride;
                const int64_t m_count = std::min(gemm_m - m_start, (int64_t) gemm_m_stride);

                const int64_t n_start = tid_n * gemm_n_stride;
                const int64_t n_count = std::min(gemm_n - n_start, (int64_t) gemm_n_stride);

                const int64_t n_blk = m_count == 1 ? n_count : NB_COLS;

                uint8_t * b_col    = reinterpret_cast<uint8_t *>(w_data) + n_start * row_stride_b;
                uint8_t * b_col_zp = block_type_has_zp<BLOC_TYPE>() ? b_col : nullptr;

                int64_t n_blk_real = 0;
                for (int64_t ni = 0; ni < n_count; ni += n_blk_real, b_col += n_blk_real * row_stride_b) {
                    n_blk_real = std::min(n_count - ni, n_blk);

                    uint8_t * a_row = quant_a + m_start * row_stride_a;

                    float * c_blk = output + m_start * gemm_n + n_start + ni;

                    int64_t rows_remaining = m_count;

                    uint8_t * b_col_cur    = b_col;
                    uint8_t * b_col_zp_cur = b_col_zp;

                    while (rows_remaining > 0) {
                        auto rows_handled = gemm_kernel(b_blk_len, a_row, b_col_cur, b_col_zp_cur, c_blk,
                                                        rows_remaining, n_blk_real, b_k_blks, gemm_n);

                        c_blk += rows_handled * gemm_n;
                        a_row += rows_handled * row_stride_a;

                        rows_remaining -= rows_handled;
                    }
                }
            }
        }
        if (spx_mm) {
            // re-derive which path the chain above actually took (the conditions only read
            // variables that are not mutated by the chain, so this is exact)
            char path_tag = 'X';
            if (gemm_m == 1 && NB_COLS == 16 && ime1_int4 && vlen256 && (mm_path == 'd' || mm_path == 'D')) {
                path_tag = 'D';
            } else if ((mm_path == 'g' || mm_path == 'G') && gemm_m > 1 && gemm_n_stride == gemm_n &&
                       tcm_buffer != nullptr &&
                       spacemit_kernels::div_round_up(spacemit_kernels::div_round_up(gemm_m, row_align), nth) *
                               per_mb_rows_wsize <=
                           tcm_buffer_size) {
                path_tag = 'G';
            } else if (prefer_tcm && gemm_n_stride == gemm_n && tcm_buffer != nullptr &&
                       per_mb_rows_wsize <= tcm_buffer_size) {
                path_tag = 'P';
            } else if ((mm_path == 'e' || mm_path == 'E') && gemm_m == 1 && tcm_buffer != nullptr &&
                       (gemm_workspace_size + slab_cols * row_stride_b) <= tcm_buffer_size) {
                path_tag = 'E';
            } else if (prefer_tcm && tcm_buffer != nullptr && slab_cols * row_stride_b <= tcm_buffer_size) {
                path_tag = 'B';
            } else if ((mm_path == 'f' || mm_path == 'F') && gemm_m > 1 && ws_slab > 0 && tcm_buffer != nullptr) {
                path_tag = 'F';
            }
            const int64_t spx_tg1 = ggml_time_us();
            if (ith == 0) {
                g_mmph.add(gemm_m, gemm_n, gemm_k, path_tag, spx_tq1 - spx_tq0, spx_tb1 - spx_tq1,
                           spx_tg1 - spx_tb1, spx_bc_us, spx_bc_bytes, spx_aread);
            }
        }
    }

    void forward_mul_mat_id(ggml_compute_params * params, ggml_tensor * op) {
        constexpr size_t a_blk_len = INTER_SIZE;
        constexpr size_t b_blk_len = INTER_SIZE;

        const ggml_tensor * src0 = op->src[0];
        const ggml_tensor * src1 = op->src[1];
        const ggml_tensor * ids  = op->src[2];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_BINARY_OP_LOCALS

        int ith = params->ith;
        int nth = params->nth;

        // row groups
        const int n_ids = ids->ne[0];  // n_expert_used
        const int n_as  = ne02;        // n_expert

        struct mmid_row_mapping {
            int32_t i1;
            int32_t i2;
        };

        spacemit_kernels::quantize_a_row_def           quantize_a_row_i8;
        spacemit_kernels::gemm_kernel_quantize_def     gemm_kernel;
        spacemit_kernels::moe_gemm_kernel_quantize_def moe_gemm_kernel_m2;
        bool                                           set_kernel_impl = false;
        size_t                                         block_stride_a  = spacemit_kernels::q8_blk_size(QK4_0);

#if defined(RISCV64_SPACEMIT_IME2)
        if (!set_kernel_impl && (global_spine_env_info.use_ime2)) {
            quantize_a_row_i8 = spacemit_kernels::rvv::quantize_a_row_i8;
            block_stride_a    = spacemit_kernels::q8_blk_size(QK4_0, true);

            if constexpr (std::is_same_v<BLOC_TYPE, block_q6_K> || std::is_same_v<BLOC_TYPE, block_q8_0>) {
                gemm_kernel     = spacemit_kernels::ime2::gemm_kernel_i8i8;
                set_kernel_impl = true;
            } else if constexpr (std::is_same_v<BLOC_TYPE, block_q4_0> || std::is_same_v<BLOC_TYPE, block_q4_1> ||
                                 std::is_same_v<BLOC_TYPE, block_q4_K>) {
                if constexpr (INTER_SIZE == 256) {
                    gemm_kernel       = spacemit_kernels::ime2::gemm_kernel_i8i4_hp;
                    quantize_a_row_i8 = spacemit_kernels::rvv::quantize_a_row_i8_hp;
                    block_stride_a    = spacemit_kernels::q8_hp_blk_size(a_blk_len, true, true);
                    set_kernel_impl   = true;
                } else {
                    gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i4;
                    moe_gemm_kernel_m2 = spacemit_kernels::ime2::moe_m2_gemm_kernel_i8i4;
                    quantize_a_row_i8  = spacemit_kernels::rvv::quantize_a_row_i8;
                    block_stride_a     = spacemit_kernels::q8_blk_size(a_blk_len, true);
                    set_kernel_impl    = true;
                }
            } else if constexpr (std::is_same_v<BLOC_TYPE, block_q2_K>) {
                quantize_a_row_i8 = spacemit_kernels::rvv::quantize_a_row_i8k;
                block_stride_a    = spacemit_kernels::q8k_blk_size(a_blk_len);
                gemm_kernel       = spacemit_kernels::ime2::gemm_kernel_i8i2k;
                set_kernel_impl   = true;
            } else if constexpr (std::is_same_v<BLOC_TYPE, block_q3_K>) {
                quantize_a_row_i8 = spacemit_kernels::rvv::quantize_a_row_i8k;
                block_stride_a    = spacemit_kernels::q8k_blk_size(a_blk_len);
                gemm_kernel       = spacemit_kernels::ime2::gemm_kernel_i8i3k;
                set_kernel_impl   = true;
            } else if constexpr (std::is_same_v<BLOC_TYPE, block_mxfp4>) {
                gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8mxfp4;
                moe_gemm_kernel_m2 = spacemit_kernels::ime2::moe_m2_gemm_kernel_i8mxfp4;
                set_kernel_impl    = true;
            } else if constexpr (std::is_same_v<BLOC_TYPE, block_q5_1> || std::is_same_v<BLOC_TYPE, block_q5_K> ||
                                 std::is_same_v<BLOC_TYPE, block_q5_0>) {
                gemm_kernel        = spacemit_kernels::ime2::gemm_kernel_i8i5;
                moe_gemm_kernel_m2 = spacemit_kernels::ime2::moe_m2_gemm_kernel_i8i5;
                set_kernel_impl    = true;
            }
        }
#endif

#if defined(RISCV64_SPACEMIT_IME1)
        if (!set_kernel_impl && (global_spine_env_info.use_ime1)) {
            quantize_a_row_i8 = spacemit_kernels::ime1::quantize_a_row_i8;

            if constexpr (std::is_same_v<BLOC_TYPE, block_q4_0> || std::is_same_v<BLOC_TYPE, block_q4_1> ||
                          std::is_same_v<BLOC_TYPE, block_q4_K> || std::is_same_v<BLOC_TYPE, block_q4_0_k64> ||
                          std::is_same_v<BLOC_TYPE, block_q4_0_k128>) {
                gemm_kernel     = spacemit_kernels::ime1::gemm_kernel_i8i4;
                set_kernel_impl = true;
            }
        }
#endif
        if (!set_kernel_impl) {
            GGML_ABORT("no kernel implementation found for the block type");
        }

        const size_t a_k_blks = spacemit_kernels::div_round_up(ne10, a_blk_len);
        const size_t b_k_blks = spacemit_kernels::div_round_up(ne10, b_blk_len);

        const size_t nbw1                = a_k_blks * block_stride_a;
        const size_t nbw2                = ne11 * nbw1;
        const size_t nbw3                = nbw2 * ne12;
        const size_t gemm_workspace_size = GGML_PAD(nbw3, alignof(int64_t));

        const uintptr_t ws_ptr         = reinterpret_cast<uintptr_t>(params->wdata);
        auto *          quant_a_buffer = reinterpret_cast<uint8_t *>(ws_ptr);

        if (ne11 == 1) {
            for (int64_t ii = ith; ii < ne12 * a_k_blks; ii += nth) {
                int64_t i12       = ii / a_k_blks;
                int64_t ak_blk_id = ii % a_k_blks;
                quantize_a_row_i8(a_blk_len, (float *) ((char *) src1->data + i12 * nb12) + ak_blk_id * a_blk_len,
                                  a_blk_len, quant_a_buffer + i12 * nbw2 + ak_blk_id * block_stride_a);
            }
        } else {
            for (int64_t ii = ith; ii < ne12 * ne11; ii += nth) {
                int64_t i12 = ii / ne11;
                int64_t i11 = ii % ne11;
                quantize_a_row_i8(a_blk_len, (float *) ((char *) src1->data + i12 * nb12 + i11 * nb11), ne10,
                                  quant_a_buffer + i12 * nbw2 + i11 * nbw1);
            }
        }

#define MMID_MATRIX_ROW(row_id, i1) matrix_rows[(row_id) *ne12 + (i1)]

        int64_t *          matrix_row_counts       = (int64_t *) (ws_ptr + gemm_workspace_size);
        int32_t *          valid_ep_count          = (int32_t *) (matrix_row_counts + n_as);
        int32_t *          valid_act_count         = (int32_t *) (valid_ep_count + 1);
        int64_t *          valid_matrix_row_counts = (int64_t *) (valid_act_count + 1);
        mmid_row_mapping * matrix_rows             = (mmid_row_mapping *) (valid_matrix_row_counts + n_as);

        if (ith == 0) {
            // initialize matrix_row_counts
            memset(matrix_row_counts, 0, n_as * sizeof(int64_t));

            // group rows by src0 matrix
            for (int32_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
                for (int32_t id = 0; id < n_ids; ++id) {
                    const int32_t i02 =
                        *(const int32_t *) ((const char *) ids->data + iid1 * ids->nb[1] + id * ids->nb[0]);

                    GGML_ASSERT(i02 >= 0 && i02 < n_as);

                    MMID_MATRIX_ROW(i02, matrix_row_counts[i02]) = { id, iid1 };
                    matrix_row_counts[i02] += 1;
                }
            }

            int32_t valid_ep_count_t  = 0;
            int32_t valid_act_count_t = 0;
            for (int cur_a = 0; cur_a < n_as; ++cur_a) {
                const int64_t cne1 = matrix_row_counts[cur_a];
                if (cne1 == 0) {
                    continue;
                }
                valid_matrix_row_counts[valid_ep_count_t] = cur_a;
                valid_act_count_t += cne1;
                valid_ep_count_t += 1;
            }
            valid_ep_count[0]  = valid_ep_count_t;
            valid_act_count[0] = valid_act_count_t;
        }

        const int64_t barrier_idx = static_cast<int64_t>(ith / 2);

        GGML_ASSERT(global_spine_env_info.init_barrier != nullptr);
        GGML_ASSERT(barrier_idx < spine_init_barrier_count);
        spine_barrier_t * cur_barrier = &global_spine_env_info.init_barrier[barrier_idx];

        ggml_barrier(params->threadpool);

        const size_t row_stride_b      = b_k_blks * get_repacked_block_type_size<BLOC_TYPE, INTER_SIZE, NB_COLS>();
        const size_t expert_b_stride   = ne01 * row_stride_b;
        const size_t per_nb_cols_wsize = NB_COLS * row_stride_b;

        std::array<const uint8_t *, 2> src_workspaces;
        std::array<float *, 2>         dst_workspaces;

        auto *     tcm_buffer      = ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer;
        const auto tcm_buffer_size = ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer_size;

        const auto valid_ep_count_t  = valid_ep_count[0];
        const auto valid_act_count_t = valid_act_count[0];

        int nth_es = 1;
        int nth_n  = nth;

        int ith_es = ith % nth_es;
        int ith_n  = (ith / nth_es) % nth_n;

        if (valid_ep_count_t % nth == 0 && tcm_buffer != nullptr && valid_ep_count_t == n_as &&
            valid_act_count_t == n_as && per_nb_cols_wsize <= tcm_buffer_size) {
            for (int64_t valid_id = ith; valid_id < valid_ep_count_t; valid_id += nth) {
                const int64_t cur_a = valid_matrix_row_counts[valid_id];

                auto * src0_cur = (uint8_t *) src0->data + cur_a * expert_b_stride;

                mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, 0);
                const int        id          = row_mapping.i1;
                const int64_t    i11         = id % ne11;
                const int64_t    i12         = row_mapping.i2;
                const int64_t    i1          = id;
                const int64_t    i2          = i12;

                auto *  src1_col = quant_a_buffer + (i11 * nbw1 + i12 * nbw2);
                float * c_blk    = (float *) ((char *) dst->data + (i1 * nb1 + i2 * nb2));

                uint8_t * a_row = src1_col;
                uint8_t * b_col = reinterpret_cast<uint8_t *>(tcm_buffer);
                if ((nbw1 + per_nb_cols_wsize) <= tcm_buffer_size) {
                    a_row = (uint8_t *) tcm_buffer;
                    b_col = reinterpret_cast<uint8_t *>(tcm_buffer) + nbw1;
                }
                uint8_t * b_col_zp = block_type_has_zp<BLOC_TYPE>() ? b_col : nullptr;

                if (ith % 2 == 0) {
                    spacemit_kernels::rvv::memcpy1d(b_col, reinterpret_cast<uint8_t *>(src0_cur), per_nb_cols_wsize);

                    if (a_row != src1_col) {
                        spacemit_kernels::rvv::memcpy1d(a_row, src1_col, nbw1);
                    }
                }

                spine_barrier_wait(cur_barrier);

                if (ith % 2 != 0) {
                    if (a_row != src1_col) {
                        spacemit_kernels::rvv::memcpy1d(a_row, src1_col, nbw1);
                    }

                    spacemit_kernels::rvv::memcpy1d(b_col, reinterpret_cast<uint8_t *>(src0_cur), per_nb_cols_wsize);
                }

                int64_t nb_real = std::min(ne01, NB_COLS);
                for (int64_t ni = 0; ni < ne01; ni += NB_COLS) {
                    if (ith % 2 != 0) {
                        spine_barrier_wait(cur_barrier);
                    }

                    gemm_kernel(b_blk_len, a_row, b_col, b_col_zp, c_blk + ni, 1, nb_real, b_k_blks, ne01);

                    if (ith % 2 == 0) {
                        spine_barrier_wait(cur_barrier);
                    }

                    const int64_t next_ni = ni + NB_COLS;
                    if (next_ni < ne01) {
                        nb_real = std::min(ne01 - next_ni, NB_COLS);
                        spacemit_kernels::rvv::memcpy1d(
                            b_col, reinterpret_cast<uint8_t *>(src0_cur) + next_ni * row_stride_b, per_nb_cols_wsize);
                    }
                }
            }
        } else {
            for (int64_t valid_id = ith_es; valid_id < valid_ep_count_t; valid_id += nth_es) {
                const int64_t cur_a = valid_matrix_row_counts[valid_id];
                const int64_t cne1  = matrix_row_counts[cur_a];

                int64_t src1_cur_start = 0;
                int64_t src1_cur_end   = cne1;

                int64_t src0_cur_start = (ith_n * ne01) / nth_n;
                int64_t src0_cur_end   = MIN(((ith_n + 1) * ne01) / nth_n, ne01);

                if (src1_cur_start >= src1_cur_end || src0_cur_start >= src0_cur_end) {
                    continue;
                }

                src0_cur_start =
                    (src0_cur_start % NB_COLS) ? src0_cur_start + NB_COLS - (src0_cur_start % NB_COLS) : src0_cur_start;
                src0_cur_end =
                    (src0_cur_end % NB_COLS) ? src0_cur_end + NB_COLS - (src0_cur_end % NB_COLS) : src0_cur_end;

                auto *    src0_cur = (uint8_t *) src0->data + cur_a * expert_b_stride + src0_cur_start * row_stride_b;
                uint8_t * b_col_zp = block_type_has_zp<BLOC_TYPE>() ? src0_cur : nullptr;

                size_t extra_tcm_buffer_size = tcm_buffer_size;
                void * extra_tcm_buffer      = tcm_buffer;
                if (tcm_buffer != nullptr && (src1_cur_end - src1_cur_start) >= 4 &&
                    (src0_cur_end - src0_cur_start) * row_stride_b <= tcm_buffer_size) {
                    spacemit_kernels::rvv::memcpy1d(tcm_buffer, src0_cur,
                                                    (src0_cur_end - src0_cur_start) * row_stride_b);
                    src0_cur = reinterpret_cast<uint8_t *>(tcm_buffer);
                    b_col_zp = block_type_has_zp<BLOC_TYPE>() ? src0_cur : nullptr;
                    extra_tcm_buffer_size -= (src0_cur_end - src0_cur_start) * row_stride_b;
                    extra_tcm_buffer = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(tcm_buffer) +
                                                                (src0_cur_end - src0_cur_start) * row_stride_b);
                }

                int ir1 = src1_cur_start;

                if (extra_tcm_buffer_size >= nbw1 && extra_tcm_buffer != nullptr) {
                    int64_t quant_a_tile_size = extra_tcm_buffer_size / nbw1;
                    do {
                        quant_a_tile_size = MIN(quant_a_tile_size, src1_cur_end - ir1);

                        uint8_t * quant_a_tile_buffer = reinterpret_cast<uint8_t *>(extra_tcm_buffer);

                        int iir1 = ir1;
                        for (; iir1 < (ir1 + quant_a_tile_size); ++iir1) {
                            mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, iir1);

                            const int id = row_mapping.i1;  // selected expert index

                            const int64_t i11 = id % ne11;
                            const int64_t i12 = row_mapping.i2;  // row index in src1

                            auto * src1_col = quant_a_buffer + (i11 * nbw1 + i12 * nbw2);
                            spacemit_kernels::rvv::memcpy1d(quant_a_tile_buffer, src1_col, nbw1);
                            quant_a_tile_buffer = quant_a_tile_buffer + nbw1;
                        }

                        quant_a_tile_buffer = reinterpret_cast<uint8_t *>(extra_tcm_buffer);
                        iir1                = ir1;

                        if (moe_gemm_kernel_m2 != nullptr) {
                            for (; iir1 < (ir1 + quant_a_tile_size - 1); iir1 += 2, quant_a_tile_buffer += 2 * nbw1) {
                                mmid_row_mapping row_mapping_0 = MMID_MATRIX_ROW(cur_a, iir1);
                                mmid_row_mapping row_mapping_1 = MMID_MATRIX_ROW(cur_a, iir1 + 1);

                                src_workspaces[0] = quant_a_tile_buffer;
                                src_workspaces[1] = quant_a_tile_buffer + nbw1;

                                dst_workspaces[0] =
                                    (float *) ((char *) dst->data + (row_mapping_0.i1 * nb1 + row_mapping_0.i2 * nb2)) +
                                    src0_cur_start;
                                dst_workspaces[1] = (float *) ((char *) dst->data +
                                                               ((row_mapping_1.i1) * nb1 + (row_mapping_1.i2) * nb2)) +
                                                    src0_cur_start;
                                moe_gemm_kernel_m2(b_blk_len, src_workspaces.data(), src0_cur, b_col_zp,
                                                   dst_workspaces.data(), 1, src0_cur_end - src0_cur_start, b_k_blks,
                                                   ne01);
                            }
                        }

                        for (; iir1 < (ir1 + quant_a_tile_size); iir1++, quant_a_tile_buffer += nbw1) {
                            mmid_row_mapping row_mapping_0 = MMID_MATRIX_ROW(cur_a, iir1);

                            gemm_kernel(
                                b_blk_len, quant_a_tile_buffer, src0_cur, b_col_zp,
                                (float *) ((char *) dst->data + (row_mapping_0.i1 * nb1 + row_mapping_0.i2 * nb2)) +
                                    src0_cur_start,
                                1, src0_cur_end - src0_cur_start, b_k_blks, ne01);
                        }

                        ir1 += quant_a_tile_size;
                    } while (ir1 < src1_cur_end);
                } else {
                    if (moe_gemm_kernel_m2 != nullptr) {
                        for (; ir1 < src1_cur_end - 1; ir1 += 2) {
                            for (int iir1 = 0; iir1 < 2; ++iir1) {
                                mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, ir1 + iir1);

                                const int id = row_mapping.i1;  // selected expert index

                                const int64_t i11 = id % ne11;
                                const int64_t i12 = row_mapping.i2;  // row index in src1

                                const int64_t i1 = id;               // selected expert index
                                const int64_t i2 = i12;              // row

                                src_workspaces[iir1] = quant_a_buffer + (i11 * nbw1 + i12 * nbw2);

                                dst_workspaces[iir1] =
                                    (float *) ((char *) dst->data + (i1 * nb1 + i2 * nb2)) + src0_cur_start;
                            }

                            moe_gemm_kernel_m2(b_blk_len, src_workspaces.data(), src0_cur, b_col_zp,
                                               dst_workspaces.data(), 1, src0_cur_end - src0_cur_start, b_k_blks, ne01);
                        }
                    }

                    for (; ir1 < src1_cur_end; ir1++) {
                        mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, ir1);

                        const int id = row_mapping.i1;  // selected expert index

                        const int64_t i11 = id % ne11;
                        const int64_t i12 = row_mapping.i2;  // row index in src1

                        const int64_t i1 = id;               // selected expert index
                        const int64_t i2 = i12;              // row

                        auto * src1_col = quant_a_buffer + (i11 * nbw1 + i12 * nbw2);

                        gemm_kernel(b_blk_len, src1_col, src0_cur, b_col_zp,
                                    (float *) ((char *) dst->data + (i1 * nb1 + i2 * nb2)) + src0_cur_start, 1,
                                    src0_cur_end - src0_cur_start, b_k_blks, ne01);
                    }
                }
            }
        }
#undef MMID_MATRIX_ROW
    }

    int repack(ggml_tensor * t, const void * data, size_t data_size) override {
        GGML_LOG_DEBUG("%s: repack tensor %s with %s_%dx%d\n", __func__, t->name, ggml_type_name(t->type),
                       (int) NB_COLS, (int) INTER_SIZE);
        return ggml::cpu::riscv64_spacemit::repack<BLOC_TYPE, INTER_SIZE, NB_COLS>(t, data, data_size);
    }
};

class tensor_traits_common : public tensor_traits_base {
    bool work_size(int n_threads, const ggml_tensor * op, size_t & size) override {
        switch (op->op) {
            case GGML_OP_FLASH_ATTN_EXT:
                {
                    const int     n_tasks = n_threads;
                    const int64_t neq2    = op->src[0]->ne[2];  // number of query heads
                    const int64_t DK      = op->src[1]->ne[0];
                    const int64_t DV      = op->src[2]->ne[0];  // DV
                    const int64_t neq1    = op->src[0]->ne[1];
                    const int64_t nek1    = op->src[1]->ne[1];

                    // Tiled flash attention scratch (tile sizes defined in common.h)
                    // Per-thread: Q_q + KQ + mask + VKQ32 + V32 + K_f32 + padding
                    // V32/K_f32 now hold every kv tile of the current head when the tiled path can
                    // run (neq1 >= GGML_FA_TILE_KV, see spx_fa_tile_q in ops.cpp), so their row
                    // extent is ceil(nek1/KV)*KV.  Must match the pointer arithmetic in
                    // ggml_compute_forward_flash_attn_ext_tiled (ops.cpp).
                    const int64_t kv_alloc = (neq1 >= GGML_FA_TILE_KV)
                                           ? ((nek1 + GGML_FA_TILE_KV - 1) / GGML_FA_TILE_KV) * GGML_FA_TILE_KV
                                           : (int64_t) GGML_FA_TILE_KV;
                    size_t prefill = sizeof(float) *
                                     (GGML_FA_TILE_Q * DK + 2 * GGML_FA_TILE_Q * GGML_FA_TILE_KV + GGML_FA_TILE_Q * DV +
                                      kv_alloc * DV + kv_alloc * DK) *
                                     n_tasks;

                    // Decode path: n_kv_chunks = n_tasks (one chunk per thread)
                    // Per-thread: VKQ accmulator (DV), partial M, partial S + intra-thread scratch for V, Q and VKQ
                    size_t n_chunks = n_tasks;
                    size_t decode   = sizeof(float) * (neq2 * n_chunks * (2 + DV) + n_tasks * (DK + 2 * DV));

                    size = MAX(prefill, decode);
                }
                return true;
            case GGML_OP_GATED_DELTA_NET:
                {
                    const int64_t S_v = op->src[2]->ne[0];
                    const int64_t K   = ggml_get_op_params_i32(op, 0);
                    // delta + sum_k + sum_q (+ state_work when K > 1) + q4 scratch
                    // (int8 k | int8 q | packed int4 state groups); see rvv_kernels.cpp
                    const int64_t q4_bytes   = (S_v / 16) * (S_v / 32) * 288 +
                                               2 * 36 * (S_v / 32);
                    const int64_t chunk_ws   = (int64_t) spacemit_kernels::rvv::gdn_scratch_floats(S_v, K, op->src[2]->ne[2]);
                    const int64_t per_thread = chunk_ws;
                    size = per_thread * sizeof(float) * n_threads;
                }
                return true;
            default:
                break;
        }
        return false;
    }

    bool compute_forward(ggml_compute_params * params, ggml_tensor * op) override {
        switch (op->op) {
            case GGML_OP_NORM:
                switch (op->src[0]->type) {
                    case GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_norm_f32(params, op);
                        return true;
                    default:
                        GGML_ABORT("fatal error");
                }
            case GGML_OP_RMS_NORM:
                switch (op->src[0]->type) {
                    case GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_rms_norm_f32(params, op);
                        return true;
                    default:
                        GGML_ABORT("fatal error");
                }
            case GGML_OP_ADD:
                switch (op->src[0]->type) {
                    case GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_binary<GGML_OP_ADD, float>(params, op);
                        return true;
                    case GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_binary<GGML_OP_ADD, _Float16>(params, op);
                        return true;
                    default:
                        ggml_compute_forward_add(params, op);
                        return true;
                }
            case GGML_OP_SUB:
                switch (op->src[0]->type) {
                    case GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_binary<GGML_OP_SUB, float>(params, op);
                        return true;
                    case GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_binary<GGML_OP_SUB, _Float16>(params, op);
                        return true;
                    default:
                        ggml_compute_forward_sub(params, op);
                        return true;
                }
            case GGML_OP_MUL:
                switch (op->src[0]->type) {
                    case GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_binary<GGML_OP_MUL, float>(params, op);
                        return true;
                    case GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_binary<GGML_OP_MUL, _Float16>(params, op);
                        return true;
                    default:
                        ggml_compute_forward_mul(params, op);
                        return true;
                }
            case GGML_OP_DIV:
                switch (op->src[0]->type) {
                    case GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_binary<GGML_OP_DIV, float>(params, op);
                        return true;
                    case GGML_TYPE_F16:
                        spacemit_kernels::rvv::forward_binary<GGML_OP_DIV, _Float16>(params, op);
                        return true;
                    default:
                        ggml_compute_forward_div(params, op);
                        return true;
                }
            case GGML_OP_FLASH_ATTN_EXT:
                forward_flash_attn_ext_f16(params, op);
                return true;
            case GGML_OP_CONT:
                {
                    const ggml_tensor * src0 = op->src[0];
                    if (op->type == src0->type && op->nb[0] != src0->nb[0] && op->nb[0] == src0->nb[1] &&
                        op->ne[3] * op->ne[2] * op->nb[2] == src0->ne[3] * src0->ne[2] * src0->nb[2]) {
                        spacemit_kernels::rvv::forward_cont_with_permute(params, op);
                    } else {
                        ggml_compute_forward_cont(params, op);
                    }
                    return true;
                }
            case GGML_OP_CPY:
                {
                    const ggml_tensor * src0 = op->src[0];
                    if (op->type == src0->type && op->nb[0] == src0->nb[1] && src0->nb[0] != src0->nb[1] &&
                        ggml_nelements(src0) == ggml_nelements(op)) {
                        spacemit_kernels::rvv::forward_cpy_with_permute(params, op);
                    } else {
                        ggml_compute_forward_cpy(params, op);
                    }
                    return true;
                }
            case GGML_OP_REPEAT:
                {
                    const bool rows_equal         = ggml_nrows(op->src[0]) == ggml_nrows(op);
                    const bool broadcast_or_equal = op->src[0]->ne[0] == 1 || op->src[0]->ne[0] == op->ne[0];

                    if (rows_equal && broadcast_or_equal) {
                        switch (op->src[0]->type) {
                            case GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_repeat_nrows<int32_t>(params, op);
                                return true;
                            case GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_repeat_nrows<int16_t>(params, op);
                                return true;
                            default:
                                break;
                        }
                    }

                    if (op->src[0]->ne[1] == 1 && op->src[0]->ne[0] == op->ne[0]) {
                        switch (op->src[0]->type) {
                            case GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_repeat_dim1<int32_t>(params, op);
                                return true;
                            case GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_repeat_dim1<int16_t>(params, op);
                                return true;
                            default:
                                break;
                        }
                    }

                    ggml_compute_forward_repeat(params, op);
                }
                return true;
            case GGML_OP_SUM_ROWS:
                {
                    if (op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32) {
                        spacemit_kernels::rvv::forward_sum_rows<float>(params, op);
                    } else {
                        ggml_compute_forward_sum_rows(params, op);
                    }
                }
                return true;
            case GGML_OP_GET_ROWS:
                {
                    if (op->src[0]->type == op->type) {
                        switch (op->src[0]->type) {
                            case GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_get_rows<int32_t>(params, op);
                                return true;
                            case GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_get_rows<int16_t>(params, op);
                                return true;
                            default:
                                break;
                        }
                    }

                    ggml_compute_forward_get_rows(params, op);
                }
                return true;
            case GGML_OP_CONCAT:
                {
                    const int32_t dim = ggml_get_op_params_i32(op, 0);
                    if (dim == 0 && op->type == op->src[0]->type) {
                        switch (op->src[0]->type) {
                            case GGML_TYPE_F32:
                                spacemit_kernels::rvv::forward_concat<int32_t>(params, op);
                                return true;
                            case GGML_TYPE_F16:
                                spacemit_kernels::rvv::forward_concat<int16_t>(params, op);
                                return true;
                            default:
                                break;
                        }
                    }

                    ggml_compute_forward_concat(params, op);
                }
                return true;
            case GGML_OP_GATED_DELTA_NET:
                switch (op->src[0]->type) {
                    case GGML_TYPE_F32:
                        spacemit_kernels::rvv::forward_gated_delta_net_f32(params, op);
                        return true;
                    default:
                        GGML_ABORT("fatal error");
                }
            default:
                break;
        }
        return false;
    }

    void forward_flash_attn_ext_f16(const ggml_compute_params * params, ggml_tensor * dst) {
        const ggml_tensor * q = dst->src[0];
        const ggml_tensor * k = dst->src[1];
        const ggml_tensor * v = dst->src[2];

        GGML_TENSOR_LOCALS(int64_t, neq, q, ne)
        GGML_TENSOR_LOCALS(size_t, nbq, q, nb)
        GGML_TENSOR_LOCALS(int64_t, nek, k, ne)
        GGML_TENSOR_LOCALS(size_t, nbk, k, nb)
        GGML_TENSOR_LOCALS(int64_t, nev, v, ne)
        GGML_TENSOR_LOCALS(size_t, nbv, v, nb)
        GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)
        GGML_TENSOR_LOCALS(size_t, nb, dst, nb)

        const int64_t DK = nek0;
        const int64_t DV = nev0;

        const bool supported_prec  = (dst->op_params[3] == GGML_PREC_F32 || dst->op_params[3] == GGML_PREC_DEFAULT);
        const bool supported_types = (q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16);
        const bool supported_shape = (DK > 0 && DK <= 128 && DV > 0 && DV <= 128);
        const bool supported_vlen  = (__riscv_vlenb() == 128);

        static const bool fa_probe_on = (getenv("LLAMA_FA_PROBE") != nullptr);

        if (!(supported_prec && supported_types && supported_shape && supported_vlen)) {
            if (fa_probe_on) {
                static int ime_probe_n = 0;
                if (ime_probe_n++ < 8) {
                    fprintf(stderr,
                            "[IMEPROBE] FALLBACK prec=%d types=%d(q=%s k=%s v=%s) shape=%d(DK=%lld DV=%lld) "
                            "vlen=%d(vlenb=%zu want 128) -> generic ggml-cpu FA\n",
                            (int) supported_prec, (int) supported_types,
                            ggml_type_name(q->type), ggml_type_name(k->type), ggml_type_name(v->type),
                            (int) supported_shape, (long long) DK, (long long) DV,
                            (int) supported_vlen, __riscv_vlenb());
                }
            }
            ggml_compute_forward_flash_attn_ext(params, dst);
            return;
        }
        if (fa_probe_on) {
            static int ime_probe_ok = 0;
            if (ime_probe_ok++ < 4) {
                fprintf(stderr, "[IMEPROBE] USING SPACEMIT IME FA KERNEL DK=%lld DV=%lld vlenb=%zu\n",
                        (long long) DK, (long long) DV, __riscv_vlenb());
            }
        }

        // total rows in q
        const int64_t nr = neq1 * neq2 * neq3;

        // rows per thread
        const int ith = params->ith;
        const int nth = params->nth;

        static constexpr int64_t Q_TILE_SZ = ggml_fa_tile_config::Q;
        const bool               use_tiled = !params->use_ref && (neq1 >= Q_TILE_SZ);

        // 4x chunks per thread
        // int     nth_scaled = nth * 4;
        // int64_t chunk_size = (nr + nth_scaled - 1) / nth_scaled;
        // int64_t nchunk     = (nr + chunk_size - 1) / chunk_size;

        // if (nth == 1 || nchunk < nth) {
        //     nchunk = nth;
        // }

        int64_t nchunk = nth;

        if (ith == 0) {
            // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
            ggml_threadpool_chunk_set(params->threadpool, nth);
        }

        ggml_barrier(params->threadpool);

        // The number of elements in each chunk
        const int64_t dr = (nr + nchunk - 1) / nchunk;

        // The first chunk comes from our thread_id, the rest will get auto-assigned.
        int current_chunk = ith;

        while (current_chunk < nchunk) {
            const int64_t ir0 = dr * current_chunk;
            const int64_t ir1 = MIN(ir0 + dr, nr);

            if (use_tiled) {
                spacemit_kernels::rvv::forward_flash_attn_ext_f16_tiled_vlen1024_vf16(
                    params, dst, ir0, ir1, ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer,
                    ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer_size);
            } else {
                spacemit_kernels::rvv::forward_flash_attn_ext_f16_one_chunk_vlen1024_vf16(
                    params, dst, ir0, ir1, ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer,
                    ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer_size);
            }

            current_chunk = ggml_threadpool_chunk_add(params->threadpool, 1);
        }
    }

    int repack(ggml_tensor * t, const void * data, size_t data_size) override {
        memcpy(t->data, data, data_size);
        return 0;
    }
};

// Impl By IME1
static const tensor_traits<block_q4_0, 32, 16>  q4_0_16x32_q8_0;
static const tensor_traits<block_q4_0_k64, 64, 16> q4_0_k64_16x64_q8_0; // [Q4_0_K64] IME1, BlkLen=64
static const tensor_traits<block_q4_0_k128, 128, 16> q4_0_k128_16x128_q8_0; // [Q4_0_K128] IME1, BlkLen=128
static const tensor_traits<block_q4_1, 32, 16>  q4_1_16x32_q8_0;
static const tensor_traits<block_q4_K, 32, 16>  q4_k_16x32_q8_0;
// Impl By IME2
static const tensor_traits<block_q2_K, 256, 32> q2_k_32x256_q8_0;
static const tensor_traits<block_q3_K, 256, 32> q3_k_32x256_q8_0;
static const tensor_traits<block_q4_0, 32, 32>  q4_0_32x32_q8_0;
static const tensor_traits<block_q4_1, 32, 32>  q4_1_32x32_q8_0;
static const tensor_traits<block_q4_0, 256, 32> q4_0_32x256_q8_0;
static const tensor_traits<block_q4_1, 256, 32> q4_1_32x256_q8_0;
static const tensor_traits<block_q4_K, 32, 32>  q4_k_32x32_q8_0;
static const tensor_traits<block_q6_K, 32, 32>  q6_k_32x32_q8_0;
static const tensor_traits<block_q8_0, 32, 32>  q8_0_32x32_q8_0;
static const tensor_traits<block_q8_0, 32, 16>  q8_0_16x32_q8_0;
static const tensor_traits<block_mxfp4, 32, 32> mxfp4_32x32_q8_0;
static const tensor_traits<block_q5_K, 32, 32>  q5_k_32x32_q8_0;
static const tensor_traits<block_q5_1, 32, 32>  q5_1_32x32_q8_0;
static const tensor_traits<block_q5_0, 32, 32>  q5_0_32x32_q8_0;
// Impl By RVV
static const tensor_traits_common               rvv_impl;

}  // namespace ggml::cpu::riscv64_spacemit

static const ggml::cpu::tensor_traits * ggml_riscv64_spacemit_get_optimal_repack_type(const ggml_tensor * cur) {
    switch (cur->type) {
        case GGML_TYPE_Q2_K:
            {
#if defined(RISCV64_SPACEMIT_IME2)
                if (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    return &ggml::cpu::riscv64_spacemit::q2_k_32x256_q8_0;
                }
#endif
            }
            break;
        case GGML_TYPE_Q3_K:
            {
#if defined(RISCV64_SPACEMIT_IME2)
                if (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    return &ggml::cpu::riscv64_spacemit::q3_k_32x256_q8_0;
                }
#endif
            }
            break;
        case GGML_TYPE_Q4_0:
            {
#if defined(RISCV64_SPACEMIT_IME2)
                if (cur->ne[1] % 32 == 0 && cur->ne[0] % 256 == 0 &&
                    (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    return &ggml::cpu::riscv64_spacemit::q4_0_32x256_q8_0;
                }

                if (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    return &ggml::cpu::riscv64_spacemit::q4_0_32x32_q8_0;
                }
#endif

#if defined(RISCV64_SPACEMIT_IME1)
                if (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    return &ggml::cpu::riscv64_spacemit::q4_0_16x32_q8_0;
                }
#endif
            }
            break;
        case GGML_TYPE_Q4_0_K64:
            {
#if defined(RISCV64_SPACEMIT_IME1)
                if (cur->ne[1] % 16 == 0 && cur->ne[0] % 64 == 0 &&
                    (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    return &ggml::cpu::riscv64_spacemit::q4_0_k64_16x64_q8_0;
                }
#endif
            }
            break;
        case GGML_TYPE_Q4_0_K128:
            {
#if defined(RISCV64_SPACEMIT_IME1)
                if (cur->ne[1] % 16 == 0 && cur->ne[0] % 128 == 0 &&
                    (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    return &ggml::cpu::riscv64_spacemit::q4_0_k128_16x128_q8_0;
                }
#endif
            }
            break;
        case GGML_TYPE_Q4_1:
            {
#if defined(RISCV64_SPACEMIT_IME2)
                // TODO
                // if (cur->ne[1] % 32 == 0 && cur->ne[0] % 256 == 0 &&
                //     (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                //     return &ggml::cpu::riscv64_spacemit::q4_1_32x256_q8_0;
                // }

                if (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    return &ggml::cpu::riscv64_spacemit::q4_1_32x32_q8_0;
                }
#endif

#if defined(RISCV64_SPACEMIT_IME1)
                if (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    return &ggml::cpu::riscv64_spacemit::q4_1_16x32_q8_0;
                }
#endif
            }
            break;
        case GGML_TYPE_Q4_K:
            {
#if defined(RISCV64_SPACEMIT_IME2)
                if (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    return &ggml::cpu::riscv64_spacemit::q4_k_32x32_q8_0;
                }
#endif

#if defined(RISCV64_SPACEMIT_IME1)
                if (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                    return &ggml::cpu::riscv64_spacemit::q4_k_16x32_q8_0;
                }
#endif
            }
            break;
        case GGML_TYPE_Q6_K:
            {
#if defined(RISCV64_SPACEMIT_IME2)
                if ((ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    return &ggml::cpu::riscv64_spacemit::q6_k_32x32_q8_0;
                }
#endif
            }
            break;
        case GGML_TYPE_Q8_0:
            {
#if defined(RISCV64_SPACEMIT_IME2)
                if ((ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    return &ggml::cpu::riscv64_spacemit::q8_0_32x32_q8_0;
                }
#endif

#if defined(RISCV64_SPACEMIT_IME1)
                // DISABLED 2026-09-15: the IME1 i8i8 M4 kernel (SQ8BitGemmM4Kernel_CompInt8_ScaleFp32_Impl)
                // traps with SIGILL on this board's silicon (illegal vmadot encoding, exit 132) during
                // prefill (m>1). q8_0 weights therefore fall back to the generic CPU kernel, which is
                // what the README op-stats table records for q8_0 (cpu_generic).
                // if (cur->ne[1] % 16 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime1)) {
                //     return &ggml::cpu::riscv64_spacemit::q8_0_16x32_q8_0;
                // }
#endif
            }
            break;
        case GGML_TYPE_MXFP4:
            {
#if defined(RISCV64_SPACEMIT_IME2)
                // TODO
                // if (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                //     return &ggml::cpu::riscv64_spacemit::mxfp4_32x32_q8_0;
                // }
#endif
            }
            break;
        case GGML_TYPE_Q5_K:
            {
#if defined(RISCV64_SPACEMIT_IME2)
                if (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    return &ggml::cpu::riscv64_spacemit::q5_k_32x32_q8_0;
                }
#endif
            }
            break;
        case GGML_TYPE_Q5_1:
            {
#if defined(RISCV64_SPACEMIT_IME2)
                if (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    return &ggml::cpu::riscv64_spacemit::q5_1_32x32_q8_0;
                }
#endif
            }
            break;
        case GGML_TYPE_Q5_0:
            {
#if defined(RISCV64_SPACEMIT_IME2)
                if (cur->ne[1] % 32 == 0 && (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2)) {
                    return &ggml::cpu::riscv64_spacemit::q5_0_32x32_q8_0;
                }
#endif
            }
            break;
        default:
            break;
    }

    return nullptr;
}

static enum ggml_status ggml_backend_riscv64_spacemit_buffer_init_tensor(ggml_backend_buffer_t buffer,
                                                                         ggml_tensor *         tensor) {
    tensor->extra =
        (void *) const_cast<ggml::cpu::tensor_traits *>(ggml_riscv64_spacemit_get_optimal_repack_type(tensor));

    GGML_UNUSED(buffer);

    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_riscv64_spacemit_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);

    void * base = buffer->context;
    if (base == nullptr) {
        return;
    }

    ggml::cpu::riscv64_spacemit::spine_mem_pool_free(base);
}

static void * ggml_backend_riscv64_spacemit_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);

    void * base = buffer->context;
    GGML_ASSERT(base != nullptr);
    return base;
}

static void ggml_backend_riscv64_spacemit_buffer_memset_tensor(ggml_backend_buffer_t buffer,
                                                               ggml_tensor *         tensor,
                                                               uint8_t               value,
                                                               size_t                offset,
                                                               size_t                size) {
    GGML_ASSERT(tensor);
    memset((char *) tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_riscv64_spacemit_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);

    void * base = buffer->context;
    GGML_ASSERT(base != nullptr);
    memset(base, value, buffer->size);
}

static void ggml_backend_riscv64_spacemit_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                            ggml_tensor *         tensor,
                                                            const void *          data,
                                                            size_t                offset,
                                                            size_t                size) {
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    auto tensor_traits = (ggml::cpu::riscv64_spacemit::tensor_traits_base *) tensor->extra;
    if (tensor_traits) {
        auto OK = tensor_traits->repack(tensor, data, size);
        GGML_ASSERT(OK == 0);
    }

    GGML_UNUSED(buffer);
}

static const ggml_backend_buffer_i ggml_backend_riscv64_spacemit_buffer_i = {
    /* .free_buffer     = */ ggml_backend_riscv64_spacemit_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_riscv64_spacemit_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_riscv64_spacemit_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_riscv64_spacemit_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_riscv64_spacemit_buffer_set_tensor,
    /* .get_tensor      = */ nullptr,
    /* .set_tensor_2d   = */ nullptr,
    /* .get_tensor_2d   = */ nullptr,
    /* .cpy_tensor      = */ nullptr,
    /* .clear           = */ ggml_backend_riscv64_spacemit_buffer_clear,
    /* .reset           = */ nullptr,
};

static const char * ggml_backend_cpu_riscv64_spacemit_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU_RISCV64_SPACEMIT";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_cpu_riscv64_spacemit_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                                                                        size_t size) {
    void * base = ggml::cpu::riscv64_spacemit::spine_mem_pool_alloc(size, 64);
    if (base == nullptr) {
        return nullptr;
    }

    return ggml_backend_buffer_init(buft, ggml_backend_riscv64_spacemit_buffer_i, base, size);
}

static size_t ggml_backend_cpu_riscv64_spacemit_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 64;

    GGML_UNUSED(buft);
}

static size_t ggml_backend_cpu_riscv64_spacemit_nbytes(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (tensor->ne[i] <= 0) {
            return 0;
        }
    }

    GGML_UNUSED(buft);

    const auto plain_nbytes = [&]() {
        size_t total = ggml_type_size(tensor->type);
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            total += (tensor->ne[i] - 1) * tensor->nb[i];
        }
        return total;
    };

    const size_t blck_size = ggml_blck_size(tensor->type);
    if (blck_size == 1) {
        return plain_nbytes();
    }

    const size_t row_nbytes = tensor->ne[0] * tensor->nb[0] / blck_size;

    const auto add_strided_nbytes = [&](size_t total, size_t src_block_size, size_t dst_block_size) {
        for (int i = 1; i < GGML_MAX_DIMS; ++i) {
            total += (tensor->ne[i] - 1) * (tensor->nb[i] / src_block_size) * dst_block_size;
        }
        return total;
    };

    const auto remap_block_nbytes = [&](size_t src_block_size, size_t dst_block_size, int64_t padded_rows = 0) {
        GGML_ASSERT(row_nbytes % src_block_size == 0);

        size_t total =
            add_strided_nbytes((row_nbytes / src_block_size) * dst_block_size, src_block_size, dst_block_size);

        if (padded_rows > 0 && tensor->ne[1] % padded_rows != 0) {
            total += (padded_rows - tensor->ne[1] % padded_rows) * (tensor->nb[1] / src_block_size) * dst_block_size;
        }

        return total;
    };

    size_t nbytes = row_nbytes;
    switch (tensor->type) {
        case GGML_TYPE_Q4_K:
            nbytes = remap_block_nbytes(sizeof(block_q4_K), sizeof(block_q4_1) * 8);
            break;
        case GGML_TYPE_Q6_K:
            nbytes = remap_block_nbytes(sizeof(block_q6_K), sizeof(block_q8_0) * 8, 32);
            break;
        case GGML_TYPE_Q8_0:
            nbytes = remap_block_nbytes(sizeof(block_q8_0), sizeof(block_q8_0), 32);
            break;
        case GGML_TYPE_Q2_K:
            nbytes = remap_block_nbytes(sizeof(block_q2_K), sizeof(spacemit_kernels::nrow_block_q2_k<1>));
            break;
        case GGML_TYPE_Q3_K:
            nbytes = remap_block_nbytes(sizeof(block_q3_K), sizeof(spacemit_kernels::nrow_block_q3_k<1>));
            break;
        case GGML_TYPE_MXFP4:
            nbytes = remap_block_nbytes(sizeof(block_mxfp4), sizeof(spacemit_kernels::nrow_block_mxfp4<1>));
            break;
        case GGML_TYPE_Q5_K:
            nbytes = remap_block_nbytes(sizeof(block_q5_K), sizeof(spacemit_kernels::nrow_block_q5_1<1>) * 8);
            break;
        case GGML_TYPE_Q5_1:
            nbytes = remap_block_nbytes(sizeof(block_q5_1), sizeof(spacemit_kernels::nrow_block_q5_1<1>));
            break;
        case GGML_TYPE_Q5_0:
            nbytes = remap_block_nbytes(sizeof(block_q5_0), sizeof(spacemit_kernels::nrow_block_q5_0<1>));
            break;
        default:
            nbytes = add_strided_nbytes(row_nbytes, 1, 1);
            break;
    }

    return nbytes;
}

namespace ggml::cpu::riscv64_spacemit {

class extra_buffer_type : ggml::cpu::extra_buffer_type {
    bool supports_op(ggml_backend_dev_t, const ggml_tensor * op) override {
        switch (op->op) {
            case GGML_OP_MUL_MAT:
                if (op->src[0]->buffer && (ggml_n_dims(op->src[0]) == 2) &&
                    op->src[0]->buffer->buft == ggml_backend_cpu_riscv64_spacemit_buffer_type() &&
                    ggml_riscv64_spacemit_get_optimal_repack_type(op->src[0])) {
                    if (op->src[1]->buffer && !ggml_backend_buft_is_host(op->src[1]->buffer->buft)) {
                        return false;
                    }
                    if (op->src[1]->type == GGML_TYPE_F32) {
                        return true;
                    }
                }
                break;
            case GGML_OP_MUL_MAT_ID:
                if (op->src[0]->buffer && (ggml_n_dims(op->src[0]) == 3) &&
                    op->src[0]->buffer->buft == ggml_backend_cpu_riscv64_spacemit_buffer_type() &&
                    ggml_riscv64_spacemit_get_optimal_repack_type(op->src[0])) {
                    if (op->src[1]->buffer && !ggml_backend_buft_is_host(op->src[1]->buffer->buft)) {
                        return false;
                    }
                    if (op->src[1]->type == GGML_TYPE_F32) {
                        return true;
                    }
                }
                break;
            default:
                // GGML_ABORT("fatal error");
                break;
        }
        return false;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(const ggml_tensor * op) override {
        switch (op->op) {
            case GGML_OP_MUL_MAT:
            case GGML_OP_MUL_MAT_ID:
                if (op->src[0]->buffer && op->src[0]->buffer->buft == ggml_backend_cpu_riscv64_spacemit_buffer_type()) {
                    return (ggml::cpu::tensor_traits *) op->src[0]->extra;
                }
                break;
            case GGML_OP_NORM:
            case GGML_OP_RMS_NORM:
            case GGML_OP_ADD:
            case GGML_OP_SUB:
            case GGML_OP_MUL:
            case GGML_OP_DIV:
            case GGML_OP_FLASH_ATTN_EXT:
            case GGML_OP_CONT:
            case GGML_OP_CPY:
            case GGML_OP_REPEAT:
            case GGML_OP_SUM_ROWS:
            case GGML_OP_GET_ROWS:
            case GGML_OP_CONCAT:
            case GGML_OP_GATED_DELTA_NET:
                return (ggml::cpu::tensor_traits *) (&ggml::cpu::riscv64_spacemit::rvv_impl);
            default:
                // GGML_ABORT("fatal error");
                break;
        }

        return nullptr;
    }
};

}  // namespace ggml::cpu::riscv64_spacemit

ggml_backend_buffer_type_t ggml_backend_cpu_riscv64_spacemit_buffer_type(void) {
    static ggml_backend_buffer_type ggml_backend_cpu_buffer_type_riscv64_spacemit = {
  /* .iface    = */
        {
         /* .get_name         = */ ggml_backend_cpu_riscv64_spacemit_buffer_type_get_name,
         /* .alloc_buffer     = */ ggml_backend_cpu_riscv64_spacemit_buffer_type_alloc_buffer,
         /* .get_alignment    = */ ggml_backend_cpu_riscv64_spacemit_buffer_type_get_alignment,
         /* .get_max_size     = */ nullptr,
         /* .get_alloc_size   = */ ggml_backend_cpu_riscv64_spacemit_nbytes,
         /* .is_host          = */ nullptr,
         },
 /* .device  = */
        ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
 /* .context = */
        new ggml::cpu::riscv64_spacemit::extra_buffer_type(),
    };

    return &ggml_backend_cpu_buffer_type_riscv64_spacemit;
}

extern "C" {
static int bind_ai_thread() {
    int  fd, bytes;
    char str[32];

    fd = open("/proc/set_ai_thread", O_WRONLY);
    if (fd < 0) {
        GGML_LOG_ERROR("try open /proc/set_ai_thread failed\n");
        return -1;
    }

    snprintf(str, 16, "%d", 0);
    bytes = write(fd, str, strlen(str));
    if (bytes < 0) {
        GGML_LOG_ERROR("try write /proc/set_ai_thread failed\n");
        close(fd);
        return -1;
    }

    close(fd);
    return 0;
}

int ggml_backend_cpu_riscv64_spacemit_num_perfer_cores(void) {
    return ggml::cpu::riscv64_spacemit::global_spine_env_info.num_perfer_cores;
}

void ggml_backend_cpu_riscv64_spacemit_set_numa_thread_affinity(int thread_n) {
    int cpu_id = sched_getcpu();
    if (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_ime2 &&
        !((1 << cpu_id) & ggml::cpu::riscv64_spacemit::global_spine_env_info.cpu_mask)) {
        GGML_PRINT_DEBUG("bind_ai_thread for thread %d, pid %d\n", thread_n, getpid());
        bind_ai_thread();
    }

    if (ggml::cpu::riscv64_spacemit::global_spine_env_info.use_tcm &&
        ggml::cpu::riscv64_spacemit::tls_context.cpu_id == -1) {
        CPU_ZERO(&(ggml::cpu::riscv64_spacemit::tls_context.cpuset));
        pthread_t    main_thread     = pthread_self();
        const auto & perfer_core_ids = ggml::cpu::riscv64_spacemit::global_spine_env_info.perfer_core_ids;
        if (thread_n < 0 || static_cast<size_t>(thread_n) >= perfer_core_ids.size()) {
            // This board exposes 8 cores but only 4 *preferred* ones, and the
            // default worker count is the core count. Aborting here killed the
            // process before a single token -- `llama-bench` with no -t died.
            // Extra threads just run unpinned and without a TCM buffer; the
            // preferred cores are still pinned, which is where the win is.
            static bool warned = false;
            if (!warned) {
                warned = true;
                GGML_LOG_WARN("%s: thread_n %d >= perfer_core_ids size %zu; "
                              "extra threads run unpinned (no TCM)\n",
                              __func__, thread_n, perfer_core_ids.size());
            }
            return;
        }
        auto perfer_cpu_id = perfer_core_ids[static_cast<size_t>(thread_n)];
        CPU_SET(perfer_cpu_id, &(ggml::cpu::riscv64_spacemit::tls_context.cpuset));
        int s =
            pthread_setaffinity_np(main_thread, sizeof(cpu_set_t), &(ggml::cpu::riscv64_spacemit::tls_context.cpuset));
        if (s != 0) {
            GGML_ABORT("set thread affinity error for thread_n %d, cpu_id %d\n", thread_n, perfer_cpu_id);
        }

        int ai_cpu_id = perfer_cpu_id - ggml::cpu::riscv64_spacemit::global_spine_env_info.aicpu_id_offset;
        ggml::cpu::riscv64_spacemit::tls_context.cpu_id = ai_cpu_id;
        ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer =
            ggml::cpu::riscv64_spacemit::spine_mem_pool_tcm_mem_get(ai_cpu_id);
        ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer_size =
            ggml::cpu::riscv64_spacemit::global_spine_env_info.tcm_blk_size;
    }

    if (ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer != nullptr) {
        void * rt =
            ggml::cpu::riscv64_spacemit::spine_mem_pool_tcm_mem_wait(ggml::cpu::riscv64_spacemit::tls_context.cpu_id);
        if (rt == nullptr) {
            GGML_ABORT("wait tcm buffer failed for cpu_id: %d", ggml::cpu::riscv64_spacemit::tls_context.cpu_id);
        }
    }
}

void ggml_backend_cpu_riscv64_spacemit_clear_numa_thread_affinity_threaded(int thread_n) {
    if (ggml::cpu::riscv64_spacemit::tls_context.tcm_buffer != nullptr) {
        auto rt = ggml::cpu::riscv64_spacemit::spine_mem_pool_tcm_mem_release(
            ggml::cpu::riscv64_spacemit::tls_context.cpu_id);
        if (rt != 0) {
            GGML_ABORT("release tcm buffer failed for cpu_id: %d", ggml::cpu::riscv64_spacemit::tls_context.cpu_id);
        }
    }
}
}
