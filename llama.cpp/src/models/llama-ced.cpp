// CED prefix K/V projection loader. See llama-ced.h for the blob format.
//
// Deliberately defensive: every failure path prints one line to stderr and
// returns an inactive ctx, so an unreadable/absent blob leaves the model
// byte-identical to stock.

#include "llama-ced.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstdarg>
#include <cstring>
#include <sys/stat.h>
#include <vector>

namespace {

const uint32_t CED_HDR_BYTES = 28;          // magic[8] + 5 * u32
const uint32_t CED_MAX_LAYERS = 4096;

// the blob is little-endian regardless of the host
uint32_t rd_u32(const unsigned char * p) {
    return ((uint32_t) p[0])       | ((uint32_t) p[1] <<  8) |
           ((uint32_t) p[2] << 16) | ((uint32_t) p[3] << 24);
}

void ced_warn(const char * fmt, ...) {
    std::fprintf(stderr, "CED: ");
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(stderr, fmt, ap);
    va_end(ap);
    std::fprintf(stderr, "\n");
}

} // namespace

void llama_ced_ctx_deleter::operator()(ggml_context * p) const {
    if (p) {
        ggml_free(p);
    }
}

void llama_ced_buf_deleter::operator()(ggml_backend_buffer * p) const {
    if (p) {
        ggml_backend_buffer_free(p);
    }
}

llama_ced_ctx llama_ced_load(const char * path, ggml_backend_buffer_type * buft) {
    llama_ced_ctx ced;   // inactive until the very end

    if (path == nullptr || path[0] == '\0') {
        ced_warn("empty projection path - feature disabled");
        return ced;
    }

    FILE * f = std::fopen(path, "rb");
    if (f == nullptr) {
        ced_warn("cannot open '%s' - feature disabled", path);
        return ced;
    }

    unsigned char hdr[CED_HDR_BYTES];
    if (std::fread(hdr, 1, CED_HDR_BYTES, f) != CED_HDR_BYTES) {
        ced_warn("'%s': truncated header - feature disabled", path);
        std::fclose(f);
        return ced;
    }
    if (std::memcmp(hdr, "CEDPROJ1", 8) != 0) {
        ced_warn("'%s': bad magic (expected CEDPROJ1) - feature disabled", path);
        std::fclose(f);
        return ced;
    }

    const uint32_t n_cal    = rd_u32(hdr +  8);
    const uint32_t hidden   = rd_u32(hdr + 12);
    const uint32_t kv_dim   = rd_u32(hdr + 16);
    const uint32_t split    = rd_u32(hdr + 20);
    // NOTE: the blob carries a `tail` field, but the exporter writes
    // max(--eval-tails) into it (see tools/ced_train_kvproj.py), i.e. the
    // largest tail it was *evaluated* at -- 512 for every shipped blob, which
    // means "skip nothing".  It is therefore NOT usable as the runtime default;
    // the deployed window is LLAMA_CED_DEFAULT_TAIL, overridable per run.
    const uint32_t tail     = rd_u32(hdr + 24);

    if (n_cal == 0 || n_cal > CED_MAX_LAYERS || hidden == 0 || kv_dim == 0) {
        ced_warn("'%s': implausible header (n_cal=%u)", path, n_cal);
        std::fclose(f);
        return ced;
    }

    std::vector<uint32_t> ids(n_cal);
    if (std::fread(ids.data(), sizeof(uint32_t), n_cal, f) != n_cal) {
        ced_warn("'%s': truncated layer id table - feature disabled", path);
        std::fclose(f);
        return ced;
    }
    for (uint32_t i = 0; i < n_cal; ++i) {
        ids[i] = rd_u32(reinterpret_cast<const unsigned char *>(&ids[i]));
    }

    // exact expected size, so a truncated or transposed export is caught here
    // ---- projection weight format ------------------------------------------
    // The projection is a plain ggml MUL_MAT.  With fp16 weights it has no
    // spacemit traits case and therefore runs in the generic CPU kernel
    // (measured 5.9 GMAC/s on K1 for this shape, ~10x below the IME1 path).
    // LLAMA_QWEN35_CED_WQ=q4_0_k128 quantises the same fp16 payload at load time
    // into the IME1-supported format and places the tensors in the CPU device's
    // extra (repacking) buffer, which is what the IME1 path requires.
    // Unset => unchanged fp16 behaviour.  The blob format is untouched.
    // LLAMA_QWEN35_CED_WQ = q4_K | q4_0_k128 | fp16 (default fp16)
    enum { CED_WQ_F16 = 0, CED_WQ_Q4K = 1, CED_WQ_Q40K128 = 2 };
    // default: q4_K (measured 2026-10-11 on K1: 3.4% faster prefill, +0.076% PPL on
    // wikitext vs fp16; q4_0_k128 is 3.5% faster but +0.139%).  LLAMA_QWEN35_CED_WQ=fp16
    // restores the original exact-fp16 behaviour.
    int ced_wq = CED_WQ_Q4K;
    {
        const char * e = std::getenv("LLAMA_QWEN35_CED_WQ");
        if (e != nullptr && std::strcmp(e, "fp16") == 0) {
            ced_wq = CED_WQ_F16;
        } else if (e != nullptr && std::strcmp(e, "q4_0_k128") == 0) {
            ced_wq = CED_WQ_Q40K128;
        }
    }
    // the quantised formats need the row length to be a whole number of blocks
    if (ced_wq != CED_WQ_F16) {
        const int64_t blck = ggml_blck_size(ced_wq == CED_WQ_Q4K ? GGML_TYPE_Q4_K : GGML_TYPE_Q4_0_K128);
        if (blck <= 0 || (int64_t) hidden % blck != 0) {
            ced_warn("'%s': hidden=%u is not a multiple of %lld - falling back to fp16 projections",
                     path, hidden, (long long) blck);
            ced_wq = CED_WQ_F16;
        }
    }
    const bool     ced_wq_q4    = ced_wq != CED_WQ_F16;
    const ggml_type wq_type     = ced_wq == CED_WQ_Q4K     ? GGML_TYPE_Q4_K
                                : ced_wq == CED_WQ_Q40K128 ? GGML_TYPE_Q4_0_K128
                                                           : GGML_TYPE_F16;
    const int64_t  wq_blck      = ced_wq_q4 ? ggml_blck_size(wq_type) : 1;
    const int64_t  wq_tsize     = ced_wq_q4 ? ggml_type_size(wq_type) : 2;
    const uint64_t wq_row_bytes = ced_wq_q4 ? (uint64_t) (hidden / wq_blck) * wq_tsize
                                            : (uint64_t) hidden * 2;
    const uint64_t wq_mat_bytes = (uint64_t) kv_dim * wq_row_bytes;

    const uint64_t mat_bytes  = (uint64_t) kv_dim * hidden * 2;   // fp16 source in the blob
    const uint64_t want_bytes = (uint64_t) CED_HDR_BYTES + 4ull * n_cal +
                                2ull * n_cal * mat_bytes;
    // Stat the file WITHOUT seeking: the payload is streamed straight after the
    // id table, and a fseek(SEEK_END) here would silently leave the read cursor
    // at EOF (that was a real bug -- it made every load fail as "truncated").
    struct stat st;
    if (::stat(path, &st) != 0) {
        ced_warn("'%s': stat failed - feature disabled", path);
        std::fclose(f);
        return ced;
    }
    if ((uint64_t) st.st_size != want_bytes) {
        ced_warn("'%s': size mismatch (got %lld, want %llu) - feature disabled",
                 path, (long long) st.st_size, (unsigned long long) want_bytes);
        std::fclose(f);
        return ced;
    }

    // one context holding 2*n_cal tensors; no_alloc = true so ggml only reserves
    // the metadata -- the data comes from the backend buffer below (this mirrors
    // what llama-model-loader / llama-adapter-lora do, and is the reason we do
    // not memcpy into tensor->data, which is NULL under no_alloc)
    const size_t arena = 2ull * n_cal * ggml_tensor_overhead() + 1024;
    ggml_init_params ip = { arena, nullptr, /*no_alloc =*/ true };
    ggml_context * ctx_raw = ggml_init(ip);
    if (ctx_raw == nullptr) {
        ced_warn("'%s': ggml_init failed - feature disabled", path);
        std::fclose(f);
        return ced;
    }
    std::shared_ptr<ggml_context> ctx(ctx_raw, llama_ced_ctx_deleter{});

    uint32_t max_id = 0;
    for (uint32_t i = 0; i < n_cal; ++i) {
        max_id = ids[i] > max_id ? ids[i] : max_id;
    }

    std::vector<llama_ced_layer_proj> layers(max_id + 1);
    std::vector<ggml_tensor *>        ordered;
    ordered.reserve(2 * n_cal);
    for (uint32_t i = 0; i < n_cal; ++i) {
        ggml_tensor * k = ggml_new_tensor_2d(ctx_raw, wq_type,
                                             (int64_t) hidden, (int64_t) kv_dim);
        ggml_tensor * v = ggml_new_tensor_2d(ctx_raw, wq_type,
                                             (int64_t) hidden, (int64_t) kv_dim);
        if (k == nullptr || v == nullptr) {
            ced_warn("'%s': tensor creation failed - feature disabled", path);
            std::fclose(f);
            return ced;
        }
        char nm[64];
        std::snprintf(nm, sizeof(nm), "ced_proj_k_%u", ids[i]);
        ggml_set_name(k, nm);
        std::snprintf(nm, sizeof(nm), "ced_proj_v_%u", ids[i]);
        ggml_set_name(v, nm);
        layers[ids[i]].k = k;
        layers[ids[i]].v = v;
        ordered.push_back(k);
        ordered.push_back(v);
    }

    if (buft == nullptr) {
        buft = ggml_backend_cpu_buffer_type();
        if (ced_wq_q4) {
            // the IME1 MUL_MAT traits fire only for tensors that live in the CPU
            // device's extra (weight-repacking) buffer type
            ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
            ggml_backend_buffer_type_t * extras = nullptr;
            if (dev != nullptr) {
                ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
                auto get_extra = (ggml_backend_dev_get_extra_bufts_t)
                    ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts");
                if (get_extra != nullptr) {
                    extras = get_extra(dev);
                }
            }
            if (extras != nullptr && extras[0] != nullptr) {
                buft = extras[0];
            } else {
                ced_warn("'%s': no CPU extra buffer type available - q4_0_k128 tensors stay in the "
                         "plain CPU buffer (correct, but the generic kernel will be used)", path);
            }
        }
    }
    ggml_backend_buffer * buf_raw = ggml_backend_alloc_ctx_tensors_from_buft(ctx_raw, buft);
    if (buf_raw == nullptr) {
        ced_warn("'%s': backend buffer allocation failed - feature disabled", path);
        std::fclose(f);
        return ced;
    }
    std::shared_ptr<ggml_backend_buffer> buf(buf_raw, llama_ced_buf_deleter{});
    // weights usage => the scheduler keeps them resident instead of re-uploading
    ggml_backend_buffer_set_usage(buf_raw, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    // stream the payload straight into the tensors
    std::vector<unsigned char> stage((size_t) mat_bytes);
    std::vector<unsigned char> qstage((size_t) wq_mat_bytes);
    std::vector<float>         frow((size_t) hidden);
    const struct ggml_type_traits * wq_traits = ced_wq_q4 ? ggml_get_type_traits(wq_type) : nullptr;
    if (ced_wq_q4 && (wq_traits == nullptr || wq_traits->from_float_ref == nullptr)) {
        ced_warn("'%s': no quantiser available for the requested CED weight format - feature disabled", path);
        std::fclose(f);
        return ced;
    }
    for (uint32_t i = 0; i < n_cal; ++i) {
        for (int which = 0; which < 2; ++which) {
            if (std::fread(stage.data(), 1, (size_t) mat_bytes, f) != mat_bytes) {
                ced_warn("'%s': truncated payload - feature disabled", path);
                std::fclose(f);
                return ced;
            }
            ggml_tensor * t = ordered[2 * i + which];
            if (!ced_wq_q4) {
                ggml_backend_tensor_set(t, stage.data(), 0, (size_t) mat_bytes);
                continue;
            }
            // fp16 (blob) -> fp32 row -> q4_0_k128, one output row at a time
            const ggml_fp16_t * src = reinterpret_cast<const ggml_fp16_t *>(stage.data());
            for (uint32_t r = 0; r < kv_dim; ++r) {
                for (uint32_t c = 0; c < hidden; ++c) {
                    frow[c] = ggml_fp16_to_fp32(src[(size_t) r * hidden + c]);
                }
                wq_traits->from_float_ref(frow.data(), qstage.data() + (size_t) r * wq_row_bytes,
                                          (int64_t) hidden);
            }
            ggml_backend_tensor_set(t, qstage.data(), 0, (size_t) wq_mat_bytes);
        }
    }
    std::fclose(f);

    ced.active      = true;
    ced.split_layer = split;
    ced.tail        = tail;
    ced.hidden_size = hidden;
    ced.kv_dim      = kv_dim;
    ced.layer_ids   = ids;
    ced.layers      = std::move(layers);
    ced.ctx         = ctx;
    ced.buf         = buf;

    std::fprintf(stderr,
            "CED: loaded %u projection layer(s) from '%s' (hidden=%u kv=%u split=%u tail=%u) "
            "wq=%s (%llu B/matrix, buf=%s)\n",
            n_cal, path, hidden, kv_dim, split, tail,
            ced_wq == CED_WQ_Q4K ? "q4_K" : ced_wq == CED_WQ_Q40K128 ? "q4_0_k128" : "fp16",
            (unsigned long long) wq_mat_bytes,
            ggml_backend_buft_name(buft));
    return ced;
}

llama_ced_ctx llama_ced_load_from_env(uint32_t n_layer, uint32_t n_embd,
                                      uint32_t n_embd_k, uint32_t n_embd_v,
                                      ggml_backend_buffer_type * buft) {
    llama_ced_ctx ced;   // inactive

    const char * env = std::getenv("LLAMA_QWEN35_CED");
    if (env == nullptr || env[0] == '\0') {
        // No blob named: stock.  Say it once, because on this board the stock
        // path is ~27% slower than the tuned one and silence reads as "fine".
        std::fprintf(stderr, "CED: LLAMA_QWEN35_CED not set -> stock path "
                             "(set it to '%s' for the tuned configuration)\n",
                     LLAMA_CED_SHIPPED_BLOB);
        return ced;                  // inactive => the graph takes the stock path
    }
    if (std::strcmp(env, "off") == 0 || std::strcmp(env, "none") == 0 ||
        std::strcmp(env, "0") == 0) {
        std::fprintf(stderr, "CED: LLAMA_QWEN35_CED=%s -> stock llama.cpp\n", env);
        return ced;
    }
    if (std::strcmp(env, "zero") == 0) {
        // CED split enabled but the prefix K/V left at zero: the control arm.
        ced.active = true;
        ced.split_layer = 0;         // 0 => llama_ced_project always declines
        ced.tail = 0;
        ced.layer_ids.clear();
        std::fprintf(stderr, "CED: LLAMA_QWEN35_CED=zero -> CED split with a "
                             "zero prefix (projection disabled)\n");
        return ced;
    }

    const char * path = env;
    if (std::strcmp(env, "1") == 0) {
        // legacy: stage 1 used LLAMA_QWEN35_CED=1 as a bare switch
        path = "/home/bianbu/k64-precision/models/ced_proj.bin";
    }

    ced = llama_ced_load(path, buft);
    if (!ced.active) {
        return ced;
    }

    // validate against the model we are about to run
    if (ced.hidden_size != n_embd || ced.kv_dim != n_embd_k || ced.kv_dim != n_embd_v) {
        ced_warn("dimension mismatch (blob %ux%u vs model n_embd=%u kv=%u) - feature disabled",
                 ced.hidden_size, ced.kv_dim, n_embd, n_embd_k);
        return llama_ced_ctx{};
    }
    if (ced.split_layer == 0 || ced.split_layer >= n_layer) {
        ced_warn("split_layer %u out of range (n_layer=%u) - feature disabled",
                 ced.split_layer, n_layer);
        return llama_ced_ctx{};
    }
    for (size_t i = 0; i < ced.layer_ids.size(); ++i) {
        if (ced.layer_ids[i] >= n_layer) {
            ced_warn("layer id %u out of range (n_layer=%u) - feature disabled",
                     ced.layer_ids[i], n_layer);
            return llama_ced_ctx{};
        }
        for (size_t j = i + 1; j < ced.layer_ids.size(); ++j) {
            if (ced.layer_ids[i] == ced.layer_ids[j]) {
                ced_warn("duplicate layer id in '%s' - feature disabled", path);
                return llama_ced_ctx{};
            }
        }
    }

    if (const char * e = std::getenv("LLAMA_QWEN35_CED_TAIL")) {
        const int t = std::atoi(e);
        if (t >= 1) {
            ced.tail = (uint32_t) t;
        }
    }
    return ced;
}

bool llama_ced_project(const llama_ced_ctx & ced, uint32_t il, ggml_context * ctx0,
                       ggml_tensor * h_prefix, ggml_tensor ** k_out, ggml_tensor ** v_out) {
    if (!ced.has(il) || h_prefix == nullptr || h_prefix->ne[0] != (int64_t) ced.hidden_size) {
        return false;
    }
    // [hidden, n_prefix] x [hidden, kv_dim]^T -> [kv_dim, n_prefix]
    *k_out = ggml_mul_mat(ctx0, ced.layers[il].k, h_prefix);
    *v_out = ggml_mul_mat(ctx0, ced.layers[il].v, h_prefix);
    return true;
}
