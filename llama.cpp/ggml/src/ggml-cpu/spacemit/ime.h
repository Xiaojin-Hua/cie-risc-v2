#pragma once

#include "ggml-alloc.h"

#ifdef __cplusplus
extern "C" {
#endif

ggml_backend_buffer_type_t ggml_backend_cpu_riscv64_spacemit_buffer_type(void);

void ggml_backend_cpu_riscv64_spacemit_set_numa_thread_affinity(int thread_n);

// Number of cores that carry the accelerated (IME/TCM) path.  The rest are
// usable but slow, so worker counts above this are a pessimization, not a
// speedup.  Returns 0 when none were detected.
int ggml_backend_cpu_riscv64_spacemit_num_perfer_cores(void);

void ggml_backend_cpu_riscv64_spacemit_clear_numa_thread_affinity_threaded(int thread_n);

void * ggml_backend_cpu_riscv64_spacemit_alloc_shared(size_t size, size_t alignment);

void ggml_backend_cpu_riscv64_spacemit_free_shared(void * ptr);

#ifdef __cplusplus
}
#endif
