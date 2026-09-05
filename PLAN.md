# PHP-CUDA — Master Plan: The Ultimate PHP CUDA Driver

## Delivery status (v0.2.0)

**Delivered and locally verified (phpize + full C/C++ syntax verification against
PHP 8.3 headers; GPU execution pending on the CI runner):**

- **Phase 1 — complete.** Build system rewritten: `config.m4` (NVCC via
  `PHP_ADD_MAKEFILE_FRAGMENT`, GPU arch auto-detection with toolkit-gated fallbacks,
  cuDNN 7/8/9 + NVTX3 + NVRTC detection), `Makefile.frag`, `compile.sh` (safe clean,
  mixed-toolchain refusal, driver/runtime compatibility check, `--uninstall`).
  All issue-#1 fixes F1–F9 implemented (F10 = reply on the issue, pending release).
- **Phase 2 — complete.** All C1–C6 bugs fixed; full core API implemented; lazy device
  init; no implicit `cudaDeviceReset`; `CudaException` + `cuda.error_mode`; ini entries;
  non-square matrix multiply with strict validation.
- **Phase 3 — substrate delivered.** `CudaTensor` with refcounted storage, zero-copy
  views (reshape/transpose/slice), broadcasting, 6 dtypes, matmul via cuBLAS
  (row-major correct), activations, softmax, reductions, in-place ops. **NVRTC**
  runtime kernel compilation + launch. Still ahead: DLPack, `fromBuffer`, concat.
- **Phase 4 — complete.** Streams, events, stream wait/query, CUDA graph
  capture/instantiate/launch. (Pinned staging buffers via `cuda_pinned_alloc`.)
- **Phase 5 — partial.** cuBLAS GEMM + batched GEMM (row-major correct), cuDNN 8/9
  convolution forward. Ahead: cuBLASLt, cuSOLVER, cuFFT, cuRAND, cuSPARSE, remaining
  cuDNN ops.
- **Phases 6–9 — partial.** Memory pool (shared-refcount lifetime safety, block reuse,
  stats), pinned/unified memory, bandwidth measurement, Dockerfile, CI matrix
  (PHP 8.1–8.4 × CUDA 11.8/12.6, compile-only + self-hosted GPU job),
  `docs/INSTALL-UBUNTU.md`, README rewrite, `composer.json`.
- **Verification so far:** phpize passes; all 6 C files pass `gcc -fsyntax-only`
  against PHP 8.3 headers; all 8 .cu files pass C++ syntax checking (4 real bugs
  found and fixed this way); all test PHP lints clean. GPU-side execution is the
  remaining gate — run `./compile.sh --test` on the target host or the Docker image.

---

## Mission

**Build the ultimate PHP CUDA driver — the brick that makes a PHPTorch possible for
whoever wants to build one. We are building a brick for a much larger wall, not the wall.**

PyTorch and JAX are ecosystems, but both stand on the same NVIDIA foundation: the CUDA
runtime/driver APIs, cuBLAS/cuBLASLt, cuDNN, cuSOLVER, NCCL, and NVRTC. This project
delivers that foundation for PHP, done correctly and completely.

### Scope contract

**IN SCOPE — the brick:**

1. A build system that works everywhere it should: Linux (glibc distros), CUDA 11.8/12.x,
   PHP 8.1–8.5, cuDNN 8/9, with optional components (NVTX, NCCL, cuSOLVER, cuFFT,
   cuSPARSE, NVML) degrading gracefully when absent.
2. Complete CUDA runtime API surface: device management, memory, streams, events,
   errors — plus CUDA Graph capture/replay.
3. The `CudaTensor` substrate: refcounted storage separate from views, strides,
   broadcasting, multi-dtype (fp32/fp16/bf16/fp64/int32/int8), zero-copy
   reshape/transpose/slice, DLPack interop, and a *minimal* op set (elementwise, matmul,
   reductions, common activations) that exists to prove the substrate — not to be an op zoo.
4. NVRTC runtime kernel compilation: custom GPU ops authored from PHP userland.
5. The NVIDIA library stack, bound with correct semantics: cuBLAS, cuBLASLt, cuDNN 9,
   cuSOLVER, cuFFT, cuRAND, cuSPARSE.
6. Memory subsystem: stream-ordered pools (`cudaMallocAsync`), pinned memory, unified
   memory with hints/prefetch.
7. Multi-GPU: device contexts, P2P, and NCCL primitives (all_reduce, broadcast,
   all_gather, reduce_scatter).
8. Observability: NVTX3 ranges, NVML metrics, event timing.
9. Production quality bar: builds with zero manual flags on the CI matrix, memory-clean
   under compute-sanitizer/valgrind, numerics verified against references, semver-stable
   API, and docs good enough that a PHPTorch author never needs to read our C source.

**OUT OF SCOPE — the wall (a future PHPTorch's job, built on this brick):**

- Autograd engines, `nn.Module`-style layer APIs, optimizers, loss functions
- Datasets, dataloaders, tokenizers, model zoos, training loops
- Distributed training orchestration (we ship NCCL primitives; orchestration is userland)
- The thousands-strong op library (the substrate + NVRTC make it writable in PHP)
- Windows support (`config.w32` is a stretch goal only; macOS has no CUDA)

**Success looks like:** a third party can `composer require php-cuda/cuda`, read only the
PHP-level docs, and build a credible PHPTorch without ever filing an issue about a
missing binding, a memory leak, or a build failure.

This document is the result of a full code review of the repository and an analysis of
[GitHub issue #1](https://github.com/l4nos/php-cuda/issues/1) ("unable to build on Ubuntu 24.04 with CUDA 12").
It contains:

- **Part 0** — Code review findings (what is broken today and why)
- **Part A** — The roadmap to make this the most complete, production-grade PHP CUDA driver possible
- **Part B** — Root-cause analysis and concrete fixes for issue #1

---

## Part 0 — Code Review Findings

### 0.1 Build system (critical — project does not compile)

| # | File | Problem |
|---|------|---------|
| B1 | `plugin/cuda.c` | `PHP_MINIT_FUNCTION` uses `CUDA_G(allow_async_operations)` and `CUDA_G(default_device)`, but the globals struct in `php_cuda.h` declares `ctx`, `enable_cpu_fallback`, `enable_memory_pool`, `batch_size`, `active_streams`, `allocated_memory`. **Hard compile error.** |
| B2 | `plugin/cuda.c` | The function table registers `cuda_set_device`, `cuda_get_device`, `cuda_device_reset`, `cuda_device_synchronize`, `cuda_memcpy_host_to_device`, `cuda_memcpy_device_to_host`, `cuda_memcpy_device_to_device` — none of these have `PHP_FUNCTION` bodies anywhere. **Link error** (`zif_*` symbols undefined). |
| B3 | `plugin/cuda.c` | `cuda_matrix_multiply_kernel_wrapper()` is declared `extern` but **defined in no translation unit**. Link error. |
| B4 | `plugin/php_cuda.h` | ~50 `PHP_FUNCTION` declarations (cuBLAS, cuDNN, streams, memory pool, multi-GPU, async, thread-safety, config) have **no implementations anywhere**. The test suite calls several of them (`cuda_cublas_create`, `cuda_cublas_gemm`, `cuda_batch_gemm`, …) and would fatal with "Call to undefined function" even if the extension linked. |
| B5 | `plugin/compile.sh` | `rm -rf .libs modules *.lo *.la *.o config.* Makefile* build libtool` — the `config.*` glob **deletes `config.m4`**, so the second run of `compile.sh` destroys the build configuration. Must whitelist generated files only (`configure`, `config.h*`, `config.log`, `config.status`, `config.nice`, …) and never `config.m4`/`config.w32`. |
| B6 | `plugin/config.m4` | `PHP_NEW_EXTENSION(cuda, cuda.c cuda_kernel.cu memory_pool.cu …)` — the PHP build system has no rule to compile `.cu` files. `Makefile.frag` exists but is never pulled in (`PHP_ADD_MAKEFILE_FRAGMENT` is missing) and references undefined `$(NVCC)`/`$(NVCC_FLAGS)`. |
| B7 | `plugin/config.m4` | `CUDA_CFLAGS="-arch=sm_30"` — **sm_30 was removed in CUDA 12**. Any CUDA 12 toolkit rejects this flag. This alone breaks every modern install (see Part B). |
| B8 | `plugin/config.m4` | NVTX check requires `nvToolsExt.h` and links `-lnvToolsExt`. On CUDA 12, NVTX3 is **header-only** (`nvtx3/nvToolsExt.h`, no library). The check fails on every current toolkit. |
| B9 | `plugin/profiler.cu/.cuh` | `#include <nvToolsExt.h>` is unconditional although NVTX is advertised as optional. Without NVTX installed, the build fails. Needs `#ifdef HAVE_NVTX` guards and the CUDA-12 `nvtx3/` path. |
| B10 | `plugin/memory_pool.cuh` | Uses `std::atomic<bool>` without `#include <atomic>`. `cuda_kernel.cu` includes this header without `<atomic>` first → compile error. |
| B11 | `plugin/config.m4` | `memory_utils.cu` and `tensor_core_ops.cu` exist in the tree but are **not in the source list** (nor in `Makefile.frag`) — dead code that rots. |
| B12 | `plugin/config.m4` | The `conftest.cu` compile runs bare `$NVCC conftest.cu` with no `-I` flags and then executes the binary on the build host — breaks cross-compiles and containers without a GPU. Should be tolerant (fall back to `nvcc --version` parsing). |
| B13 | `plugin/compile.sh` | `find_cuda()` only probes `/usr/local/cuda*`, `/opt/cuda`, `/usr/cuda`. Debian/Ubuntu `nvidia-cuda-toolkit` installs under `/usr/lib/cuda` + `/usr/include`; WSL and conda installs differ again. No `CUDNN_PATH` auto-detection either. |
| B14 | `plugin/compile.sh` | No detection of PHP version/ZTS/NTS, no check that `phpize` and `php-config` belong to the same PHP, no `--uninstall`, no `make test` wiring to the `.phpt` suite (README documents a `tests/run-test.php` that does not exist). |

### 0.2 Correctness bugs (runtime)

| # | File | Problem |
|---|------|---------|
| C1 | `plugin/cuda.c` `cuda_free` | Calls `cudaFree()` explicitly **and then** `zend_list_close()`, whose destructor (`cuda_memory_dtor`) calls `cudaFree()` again → **double free**. The destructor should own the free; `cuda_free` should only close the resource. |
| C2 | `plugin/cuda.c` `cuda_matrix_multiply` | Dimension math is wrong: `k` is set to `rows(B)` and the result is built as `m × k`, and `host_b` is sized `n*k`. Correct: `A(m×n) * B(n×p) = C(m×p)`. It only works for square matrices (which is all the test uses). Also missing the `cols(A) == rows(B)` validation. |
| C3 | `plugin/cuda.c` `cuda_matrix_multiply` | `cudaError_t error;` is declared mid-function and read at the bottom (`if (error != cudaSuccess)`) even on early `goto cleanup` paths where it was never assigned → **undefined behavior**. |
| C4 | `plugin/cuda.c` | `arginfo` mismatches: `cuda_free` uses `arginfo_device` (a device-id signature for a resource parameter); `cuda_get_error_string` is registered with `arginfo_void` but requires one parameter. Breaks reflection and arg checking. |
| C5 | `plugin/cuda.c` | `ZEND_HASH_FOREACH_NUM_KEY_VAL` keys are `zend_ulong`; the code uses `int i, j` → truncation on large matrices. |
| C6 | `plugin/cuda.c` MINIT/MSHUTDOWN | `cudaSetDevice(0)` in MINIT makes the **extension unloadable on GPU-less machines** (e.g. web servers without GPUs, CI). `cudaDeviceReset()` in MSHUTDOWN nukes the CUDA context for the whole process — dangerous under php-fpm where other code may share it. Initialization must be lazy; reset must be opt-in. |
| C7 | `plugin/neural_net.cu` `cuda_create_linear_layer` | `cuda_tensor_create(&(*layer)->grad_input, 1, weight_dims, …)` — `ndims=1` with a 2-element `weight_dims` array → **out-of-bounds read**. |
| C8 | `plugin/neural_net.cu` `forward_linear` | PHP data is row-major; cuBLAS is column-major. The `cublasSgemm` call treats row-major buffers as column-major → **wrong results** unless the transpose trick is applied deliberately (it isn't). Same class of bug in `matrix_ops.cu` (`cublasSgemmBatched` argument order). |
| C9 | `plugin/neural_net.cu` | `cuda_model_add_layer`, `cuda_model_backward`, `cuda_model_update`, `cuda_model_load`, `cuda_create_maxpool_layer`, `cuda_create_batchnorm_layer` are declared but never defined; `LAYER_RELU` forward uses `layer->output` which is never allocated for ReLU layers → null deref. No weight initialization (Xavier/He) anywhere. |
| C10 | `plugin/conv_ops.cu` | Uses `cudnnGetConvolutionForwardAlgorithm` — **removed in cuDNN 9** (deprecated in 8). Every cuDNN return value is ignored. The `stream` parameter is accepted but never applied (`cudnnSetStream` never called). Padding is hardcoded to 1. |
| C11 | `plugin/tensor_ops.cu` | `cuda_tensor_sigmoid`, `cuda_tensor_tanh`, `cuda_tensor_multiply`, `cuda_tensor_scale`, `cuda_tensor_softmax`, `cuda_tensor_reshape` and three backward ops are declared but not implemented. Grid size computed in `int` from `size_t total_size` → overflow beyond ~8M elements. No dtype dispatch (only float paths exist despite a `dtype` field). |
| C12 | `plugin/memory_pool.cu` | `cuda_memory_pool_resize` / `cuda_memory_pool_defragment` are empty stubs; freed blocks are never coalesced, so the pool fragments exactly as a naive free-list does. Lock ordering (pool mutex + per-block atomics) is redundant — the mutex alone suffices. |
| C13 | `plugin/cuda_utils.cuh` vs `plugin/php_cuda.h` | `CUDA_CHECK_ERROR` is **defined twice with different semantics** (one returns `void`, one returns the error) → macro redefinition errors when both headers meet. |
| C14 | `plugin/profiler.cu` | `cuda_memory_get_peak_usage`, `cuda_get_device_utilization`, `cuda_get_memory_utilization`, `cuda_get_kernel_metrics` are stubs returning success with no data. NVML/CUPTI integration is missing. |
| C15 | `plugin/cpu_ops.cu` | OpenMP pragmas compile silently without `-fopenmp`; no `HAVE_OPENMP` guard, no non-OMP fallback annotation. `cpu_convolution` ignores multiple output channels (single filter only). |
| C16 | Repo | README contains literal placeholder text (`[Rest of the README content remains unchanged...]`), a `yourusername` clone URL, and documents APIs that don't exist. `LICENSE.MD` is a ~10 KB file that is not the MIT license text (verify/replace). No `config.w32` (no Windows path at all). No CI. |

### 0.3 Design gaps (for "real world use")

- **No tensor object in PHP-land.** Real users want `$a = CudaTensor::fromArray($data)->relu()->matmul($b)`, not manual `cuda_malloc`/`cuda_memcpy` juggling with raw resources.
- **No dtype support** beyond implicit float32 (no fp16/bf16/int8/fp64), despite Tensor Core code existing.
- **No stream/event PHP API implemented** (declared only), so no async pipelining — the single biggest real-world performance lever.
- **No device-memory-backed buffer type** that survives across calls; every op round-trips PHP arrays through host memory (dominates runtime for anything non-trivial).
- **No error model**: everything is `E_WARNING` + `false`. Real-world PHP needs a `CudaException` hierarchy.
- **No php.ini settings** despite globals existing (`enable_cpu_fallback`, `batch_size`, …).
- **ZTS/thread-safety story undefined** (globals accessed without locks; `pthread_mutex_t` used but ZTS builds not handled).

---

## Part A — Roadmap to the Ultimate PHP CUDA Driver

Phases are ordered: each phase leaves the tree buildable and testable. P0 items are release blockers.

### Phase 1 — Make it build everywhere (P0) — *fixes issue #1, see Part B*

1. Reconcile module globals between `php_cuda.h` and `cuda.c`; add missing `PHP_FUNCTION` bodies for everything in the function table; remove or implement every declared function (no orphan declarations).
2. Rewrite `config.m4`: `.cu` compilation via `PHP_ADD_MAKEFILE_FRAGMENT` + working NVCC rules; architecture flags computed from detected GPUs (`native`) with a sane default list (`sm_70;sm_75;sm_80;sm_86;sm_89;sm_90;sm_120` gated by toolkit version); drop `sm_30`.
3. Version-gate cuDNN (7/8/9 API differences) and NVTX (nvToolsExt vs header-only nvtx3) with `#if CUDNN_MAJOR` / `HAVE_NVTX` guards.
4. Fix `compile.sh` (no `config.m4` deletion, toolchain detection, PHP version checks) and add a `Dockerfile` (Ubuntu 24.04 + PHP 8.4 + CUDA 12.x) as the reference environment.
5. CI (GitHub Actions): build matrix PHP {8.1–8.4} × CUDA {11.8, 12.x} in GPU-less containers (compile-only) + self-hosted GPU runner for `.phpt` execution.

### Phase 2 — Core runtime correctness (P0)

1. Fix C1–C6 (double free, matrix dims, uninitialized error, arginfo, key types, lazy init).
2. Single error-handling strategy: one `CUDA_CHECK_*` macro set; a `CudaException` thrown from PHP-facing failures (opt-in legacy warning mode via ini).
3. Implement the actually-registered core API completely: device management, `cuda_memset`, all three memcpy directions (with PHP string/array sources), `cuda_get_error_name`.
4. Lazy device init + per-request cleanup that never calls `cudaDeviceReset` implicitly.
5. php.ini entries: `cuda.enable`, `cuda.default_device`, `cuda.error_mode`, `cuda.memory_pool`, `cuda.cpu_fallback`.
6. Expand `001-basic.phpt` to cover non-square matrices, oversized inputs, and error paths; make the whole existing suite pass.

### Phase 3 — First-class `CudaTensor` object (P1) — *the ATen/c10 substrate*

This is the layer a PHPTorch would build on. It must match the memory model of a real
tensor library, not just expose malloc/free:

1. New `CudaTensor` PHP class (C-level object) holding: device pointer, shape, **strides**,
   dtype, device id, owning stream, and a **refcounted storage object** separate from the
   view (so `reshape`/`transpose`/`slice` are zero-copy views sharing one allocation —
   exactly how ATen works).
2. Constructors: `CudaTensor::fromArray()`, `::zeros()`, `::ones()`, `::rand()`, `::full()`,
   `::fromBuffer()` (zero-copy from PHP string / FFI pointer), `::fromDLPack()`.
3. Core op set (dtype-dispatched kernels + cuBLAS where applicable): `add/sub/mul/div`
   (with broadcasting), `matmul`, `transpose`, `reshape`, `slice`, `concat`, `sum/mean/max/min`
   reductions, `exp/log/sqrt/pow`, comparison ops, `relu/sigmoid/tanh/gelu/softmax`.
   This is the *minimum viable op set* — enough to prove the substrate, not an op zoo.
4. Dtypes: fp32 first, then fp16/bf16 (Tensor Core paths already sketched in
   `tensor_core_ops.cu`), fp64, int32/int8.
5. In-place variants (`add_`, `mul_`) with correct aliasing behavior on shared storage.
6. **NVRTC integration**: `cuda_kernel_compile($cudaCSource, $kernelName)` compiles PTX at
   runtime and returns a launchable kernel handle; `CudaTensor` pointers can be passed as
   kernel arguments. This is what lets PHP userland author custom ops — the single most
   important "platform" feature.
7. Interop: DLPack export/import for zero-copy exchange with other runtimes; documented
   buffer layout for `fromNumpyLikeBuffer()`.

### Phase 4 — Streams, events, CUDA Graphs (P1) — *the execution model*

1. Implement the declared stream API: `cuda_stream_create/destroy/synchronize`,
   `cuda_stream_wait_event`, `cuda_stream_query`, `cuda_event_*` exposure to PHP.
2. Async memcpy + async kernel entry points returning "future" resources; `CudaTensor`
   operations accept an optional stream.
3. **CUDA Graph capture/replay** (`cuda_stream_begin_capture`, `cuda_graph_instantiate`,
   `cuda_graph_launch`): capture a sequence of `CudaTensor` ops once, replay it with
   near-zero CPU overhead. This is the 2026-era performance baseline for repeated
   inference/training step shapes and a hard requirement for serious use.
4. Pinned-memory staging buffers for async H2D/D2H overlap; document the double-buffering
   pattern.

### Phase 5 — The NVIDIA library stack (P1) — *what PyTorch actually links against*

1. **cuBLAS / cuBLASLt**: implement every declared cuBLAS function (handle resources,
   `gemm`, `gemmEx`, batched gemm, `axpy`, `dot`, `gemv`) with correct
   row-major↔column-major handling. Add **cuBLASLt** — the modern matmul library PyTorch
   actually uses: fused epilogues (bias + activation), fp16/bf16/int8, workspace and
   heuristic-based algo selection.
2. **cuDNN 9**: conv forward/backward (data+filter), pooling fwd/bwd, activation fwd/bwd,
   batchnorm fwd/bwd, softmax, dropout — using the cuDNN 8/9 algorithm discovery and
   (where beneficial) the graph API, with workspace management and stream wiring.
3. **cuSOLVER** (`--with-cusolver`): LU/QR/Cholesky/SVD/eigendecomposition — required for
   anything beyond inference (least squares, PCA, linear algebra research).
4. **cuFFT** (`--with-cufft`): 1D/2D/3D FFTs — signal processing and spectral methods.
5. **cuRAND**: Philox-based RNG states on device; `CudaTensor::rand*()` backed by it.
6. **cuSPARSE** (`--with-cusparse`, P2): sparse CSR/CSC/COO tensors and SpMV/SpGEMM —
   needed for embeddings and GNN-style workloads.
7. Fix `neural_net.cu` (C7–C9) or delete it in favor of the `CudaTensor` substrate
   (preferred — avoids a parallel object model). Numeric tests verify every binding
   against PHP reference implementations.

### Phase 6 — Memory subsystem (P2)

1. Finish the pool: size-bucketed free lists, block splitting/coalescing, real `defragment`, per-device pools, stats exposure; or adopt `cudaMallocAsync` stream-ordered pools on CUDA 11.2+ (less code, better semantics) with the custom pool as fallback.
2. Unified memory + pinned memory PHP API built on `memory_utils.cu` (add it to the build); prefetch/advise hints exposed on `CudaTensor`.
3. Optional leak checker (debug ini flag) that reports live device allocations at request end.

### Phase 7 — Multi-GPU and NCCL (P2) — *the "scalable" part*

1. Device context registry (the `cuda_context` struct already sketched): per-device
   streams, handles, pools; peer-access enablement; P2P copies.
2. `cuda_get_optimal_device` (free-memory/utilization based), `CudaTensor::toDevice($id)`,
   `cuda_gather` helpers.
3. **NCCL bindings** (`--with-nccl`): communicator init, `all_reduce`, `broadcast`,
   `all_gather`, `reduce_scatter` over `CudaTensor` storages on multiple devices — the
   same primitives `torch.distributed` is built on. Multi-node via NCCL's bootstrap
   interface documented for worker deployments.
4. ZTS audit: lock ordering documented, globals made thread-local-safe, php-fpm/ZTS and
   persistent-worker (RoadRunner/Octane) smoke tests.

### Phase 8 — Observability (P2)

1. NVTX3 ranges around every public API (compiled out when disabled).
2. NVML backend (`--with-nvml`) for utilization/temperature/power/clock metrics; CUPTI (`--with-cupti`) for kernel-level metrics — replacing the current stubs.
3. Event-based per-op timing on `CudaTensor` (`->time(fn)` helper) and a `cuda_benchmark()` utility.

### Phase 9 — Packaging, docs, ecosystem (P1, ongoing)

1. Composer package (`php-cuda/cuda`) with a `Cuda\` PSR-4 helper library (ndarray-style PHP sugar over the C API), version constraint checking the extension.
2. Full README rewrite (remove placeholder sections), `docs/` with: installation per-distro (Ubuntu 22.04/24.04, Debian, RHEL, WSL2), API reference generated from stubs, cookbook (image batch inference, linear regression training, Laravel queue worker pattern).
3. Prebuilt binaries via GitHub Releases for common PHP/CUDA combos; `pie` (PHP Installer for Extensions) support.
4. Fuzz the PHP↔C boundary (random shapes/dtypes), valgrind/compute-sanitizer CI job, API stability policy + semver.

---

## Part B — Issue #1: "unable to build on Ubuntu 24.04 with CUDA 12"

**Reporter environment:** Ubuntu 24.04, PHP 8.4.10 (NTS), Quadro T2000 (compute 7.5), driver 535.230.02 (max CUDA runtime 12.2), `nvcc` 12.0, `cuda-toolkit-12-9` + cuDNN 9 packages installed side-by-side.

### Root causes (why the build fails for them)

1. **`-arch=sm_30` (B7):** CUDA 12 removed Kepler support; `nvcc` 12.x errors with `Unsupported gpu architecture 'compute_30'`. Guaranteed failure on any CUDA 12 toolkit.
2. **cuDNN 9 (C10):** `cudnnGetConvolutionForwardAlgorithm` was removed in cuDNN 9; `conv_ops.cu` fails to compile against `cudnn9-cuda-12`.
3. **NVTX (B8/B9):** `cuda-nvtx-12-9` ships header-only NVTX3 under `nvtx3/`; `config.m4`'s `-lnvToolsExt` and `profiler.cu`'s `#include <nvToolsExt.h>` both fail.
4. **Source-level breakage (B1–B3, B10):** even with correct flags, the tree cannot compile/link (globals mismatch, missing function bodies, missing `<atomic>`).
5. **Environment confusion (their side, but we must handle it):** they installed **both** `cuda-toolkit-12-9` (NVIDIA repo) and `nvidia-cuda-toolkit` (Ubuntu repo) plus three different cuDNN packages; `nvcc --version` reports 12.0 while toolkit 12.9 is installed → PATH points at a stale/mixed toolchain. Also, driver 535 caps the runtime at CUDA 12.2, so binaries built against the 12.9 runtime may fail at load time unless `cuda-compat` is used — the build system should detect and warn about this.
6. **`compile.sh` self-destruction (B5):** if they ran it twice, `config.m4` was deleted, producing very confusing subsequent errors.

### Fixes to implement (mapped to the plan)

| Fix | Where | Detail |
|-----|-------|--------|
| F1 | `config.m4` | Compute `-gencode` flags from `nvidia-smi --query-gpu=compute_cap` (fallback list `70;75;80;86;89;90`; add `100;120` when `CUDART_VERSION >= 12080`). Never emit `sm_30`. Honor a `--with-cuda-arch=` override. |
| F2 | `conv_ops.cu`, `neural_net.cu` | `#if CUDNN_MAJOR >= 9` → use `cudnnGetConvolutionForwardAlgorithm_v7`/`cudnnFindConvolutionForwardAlgorithm`; `#elif CUDNN_MAJOR >= 8` → `_v7` heuristics; else legacy. Same gating for any other removed APIs. |
| F3 | `config.m4`, `profiler.*` | NVTX detection order: `nvtx3/nvToolsExt.h` (header-only, define `PHP_CUDA_NVTX3`, link nothing) → `nvToolsExt.h` + `-lnvToolsExt` (legacy). Guard all NVTX code with `#ifdef HAVE_NVTX`; build must succeed without it. |
| F4 | `cuda.c`, `php_cuda.h` | Fix B1–B4: align globals, implement all registered functions, remove undeclared/unimplemented entries from the header or implement them. |
| F5 | `memory_pool.cuh` | Add `#include <atomic>` (self-contained headers policy: every header compiles standalone — add a CI check that compiles each header alone). |
| F6 | `compile.sh` | Remove `config.*` from the clean glob (delete only `configure config.h config.h.in~ config.log config.status config.nice autom4te.cache`); detect mixed toolchains by comparing `nvcc --version` against `$CUDA_DIR/version.json`; refuse to continue with a clear message. |
| F7 | `compile.sh`/`config.m4` | Driver/runtime compatibility check: parse `nvidia-smi` "CUDA Version: X.Y" and warn/fail when the toolkit runtime exceeds the driver's supported runtime, suggesting `cuda-compat-12-x` or a driver upgrade. |
| F8 | Docs | New `docs/INSTALL-UBUNTU.md` with the exact minimal package set for Ubuntu 24.04 (`cuda-toolkit-12-x`, `libcudnn9-dev-cuda-12`, `php-dev`, **not** `nvidia-cuda-toolkit`, which conflicts), PATH setup, and a verification one-liner (`php -m \| grep cuda && php -r 'var_dump(cuda_device_count());'`). |
| F9 | Repo | Add the Ubuntu 24.04 + PHP 8.4 + CUDA 12.x Dockerfile so this exact environment is reproducible and CI-tested. |
| F10 | Reply on issue | After F1–F9 land, post the resolution: explanation of sm_30/cuDNN9/NVTX3 fixes, the corrected package list, and the driver 535 vs toolkit 12.9 caveat. |

### Definition of done for issue #1

On a clean Ubuntu 24.04 container with PHP 8.4-dev, `cuda-toolkit-12-9`, and `libcudnn9-dev-cuda-12`:
`./compile.sh` completes, `php -m` shows `cuda`, and `make test` passes `001-basic.phpt` and `008-cublas.phpt` on a GPU runner — with no manual flag editing.

---

## Suggested execution order

1. **Week 1:** Phase 1 + Part B fixes (F1–F10) → v0.2.0 "it builds and runs the basic suite".
2. **Week 2–3:** Phase 2 (correctness) → v0.3.0.
3. **Week 4–7:** Phase 3 (`CudaTensor`) → v0.4.0 — the headline feature.
4. **Week 8–10:** Phases 4–5 (streams/async, cuBLAS/cuDNN) → v0.5.0.
5. **Then:** Phases 6–9 iteratively → v1.0.0 when the full `.phpt` suite, sanitizers, and the PHP 8.1–8.4 × CUDA 11.8/12.x matrix are green.
