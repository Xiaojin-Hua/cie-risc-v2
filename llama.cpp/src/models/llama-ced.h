// CED (Causal Encoder-Decoder) prefix K/V projection support for qwen35.
//
// Stage 2 of the Tail-Replay prefill pruning. The encoder half (layers
// 0..split-1) runs on every prompt token; the decoder half (layers split..)
// runs only on the trailing `tail` tokens. The prefix K/V of the full-attention
// decoder layers is reconstructed from the layer-(split-1) hidden states by a
// learned linear map P_l: H_split -> K_l / V_l, produced on the GPU side by
//   results/2026-09-29-k1-pop-bridge/tools/ced_train_kvproj.py --export
// and shipped as ONE flat binary blob, so the board needs no numpy, no extra
// GGUF and no new build dependency:
//
//   offset 0   : char magic[8]  = "CEDPROJ1"
//   offset 8   : uint32 n_layers_calibrated
//   offset 12  : uint32 hidden_size          (2048)
//   offset 16  : uint32 kv_dim               (512)
//   offset 20  : uint32 split_layer          (14)
//   offset 24  : uint32 tail                 (tail the trainer assumed)
//   offset 28  : uint32 layer_ids[n_layers_calibrated]
//   then, per calibrated layer, in layer_ids order:
//       fp16  K[kv_dim * hidden_size]   row-major, ne[0] = hidden_size (input dim)
//       fp16  V[kv_dim * hidden_size]
//
// All little-endian fp16, which is what the board consumes directly.
//
// Env:
//   LLAMA_QWEN35_CED=<path>       enable; value is the blob path. "1" means
//                                 "use the default path". Unset => inert.
//   LLAMA_QWEN35_CED_TAIL=<n>     optional override of the blob's tail
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

struct ggml_context;
struct ggml_backend_buffer;
struct ggml_backend_buffer_type;
struct ggml_tensor;

struct llama_ced_layer_proj {
    ggml_tensor * k = nullptr;   // [n_embd, n_embd_k]
    ggml_tensor * v = nullptr;   // [n_embd, n_embd_v]
};

// Deleters, so the loader can own its context/buffer without pulling
// ggml-backend.h into every translation unit that includes this header.
struct llama_ced_ctx_deleter { void operator()(ggml_context *        p) const; };
struct llama_ced_buf_deleter   { void operator()(ggml_backend_buffer * p) const; };

struct llama_ced_ctx {
    bool     active      = false;
    uint32_t split_layer = 0;
    uint32_t tail        = 0;

    // raw header fields, kept so llama_ced_load_from_env() can validate the
    // blob against the model it is about to run
    uint32_t hidden_size = 0;
    uint32_t kv_dim      = 0;
    std::vector<uint32_t> layer_ids;            // calibrated ids, in blob order

    std::vector<llama_ced_layer_proj> layers;   // indexed by absolute layer id

    // own the arena + backend buffer so the weights outlive the loader
    std::shared_ptr<ggml_context>        ctx;
    std::shared_ptr<ggml_backend_buffer> buf;

    bool has(uint32_t il) const {
        return active && il < layers.size() && layers[il].k != nullptr && layers[il].v != nullptr;
    }
};

// Read `path` into freshly allocated ggml tensors. The tensors are allocated on
// `buft` (NULL => ggml_backend_cpu_buffer_type()) and filled with
// ggml_backend_tensor_set, so this works with contexts created with
// no_alloc = true, which is how every model-weight context in this tree is made.
// Returns an INACTIVE ctx on any problem: no throw, no abort, model unchanged.
llama_ced_ctx llama_ced_load(const char * path,
                             ggml_backend_buffer_type * buft = nullptr);

// The tuned configuration is the DEFAULT; LLAMA_QWEN35_CED is an override:
//
//   unset / ""    -> the shipped best blob (see LLAMA_CED_DEFAULT_BLOB)
//   off | none|0  -> inactive: stock llama.cpp, no CED at all
//   zero          -> active but with projection disabled (zero-prefix control)
//   1             -> legacy bare switch (the older ridge blob)
//   <path>        -> that blob
//
// Any problem (missing file, bad magic, dimension mismatch) leaves the ctx
// INACTIVE so the model runs stock -- a broken blob must never be fatal.
#define LLAMA_CED_DEFAULT_BLOB "/home/bianbu/k64-precision/models/ced_proj_vB.bin"
#define LLAMA_CED_DEFAULT_TAIL 128

llama_ced_ctx llama_ced_load_from_env(uint32_t n_layer,
                                      uint32_t n_embd,
                                      uint32_t n_embd_k,
                                      uint32_t n_embd_v,
                                      ggml_backend_buffer_type * buft = nullptr);

// Multiplies the prefix part of the layer-13 hidden state by the projection of
// layer `il`. Returns false when the layer has no calibrated projection (in
// which case the caller keeps the stock/zero behaviour for that layer).
bool llama_ced_project(const llama_ced_ctx &   ced,
                       uint32_t                il,
                       struct ggml_context *   ctx0,
                       struct ggml_tensor *    h_prefix,   // [n_embd, n_prefix]
                       struct ggml_tensor **   k_out,      // [n_embd_k, n_prefix]
                       struct ggml_tensor **   v_out);     // [n_embd_v, n_prefix]
