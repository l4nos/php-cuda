# PHP CUDA Extension

A CUDA driver for PHP: device-resident tensors, the NVIDIA library stack
(cuBLAS, cuDNN, cuRAND-class ops), streams, CUDA graphs, and NVRTC runtime
kernel compilation: the foundation layer that makes a "PHPTorch" possible.

This project deliberately builds the **brick, not the wall**: autograd,
`nn.Module`, optimizers and training loops belong to frameworks built *on top*
of this driver.

## Features

- **`CudaTensor`**: device-resident n-dimensional arrays
  - Refcounted storage with zero-copy views (`reshape`, `transpose`, `slice`)
  - Broadcasting elementwise ops, `matmul` (cuBLAS), activations, softmax, reductions
  - dtypes: fp32, fp64, int32, fp16, bf16, int8
- **NVRTC**: compile and launch custom CUDA kernels from PHP at runtime
- **Streams, events, CUDA graphs**: async execution and capture/replay
- **cuBLAS**: GEMM, batched GEMM (correct row-major handling)
- **cuDNN 8/9**: convolution forward (version-gated API)
- **Memory**: device/pinned/unified allocations, growing memory pool with block reuse
- **Multi-GPU**: device enumeration, switching, per-device tensors
- **Error model**: `E_WARNING` + `false` by default, `CudaException` mode via `cuda.error_mode=exception`

## Requirements

- PHP 8.1 – 8.5 (NTS or ZTS)
- CUDA Toolkit 11.8 or newer (12.x recommended)
- GPU with compute capability 7.0+ (Volta or newer; older archs down to 5.0 build with `--with-cuda-arch`)
- cuDNN 8 or 9 (optional but recommended)
- Linux (glibc). Windows/macOS are not supported (macOS has no CUDA).

## Installation

```bash
cd plugin
./compile.sh            # auto-detects CUDA, cuDNN, GPU architectures
./compile.sh --test     # build + run the test suite
```

Manual build:

```bash
phpize
./configure --with-cuda=/usr/local/cuda --with-cudnn=/usr/local/cuda --with-nvrtc=/usr/local/cuda
make -j$(nproc)
sudo make install
echo "extension=cuda.so" | sudo tee "$(php-config --ini-dir)/cuda.ini"
```

See [docs/INSTALL-UBUNTU.md](docs/INSTALL-UBUNTU.md) for the full Ubuntu 24.04
guide (including the exact apt package set and common pitfalls), or use the
reference [Dockerfile](Dockerfile).

## Quick start

```php
<?php
// Tensors live on the GPU; host transfers happen only at the edges.
$a = CudaTensor::fromArray([[1.0, 2.0], [3.0, 4.0]]);
$b = CudaTensor::full([2, 2], 0.5);

$c = $a->matmul($b)->relu()->add(1.0);
print_r($c->toArray());

// Broadcasting, views, reductions
$row = CudaTensor::fromArray([10.0, 20.0]);
echo $a->add($row)->sum(), "\n";       // 40
$t = $a->transpose();                   // zero-copy view
print_r($t->shape());                   // [2, 2]

// Custom kernels, compiled at runtime (NVRTC)
$src = '
extern "C" __global__ void scale2(float* x, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= 2.0f;
}';
$kernel = cuda_kernel_compile($src, 'scale2');
cuda_kernel_launch($kernel, [$a, 4], [1], [256]);
cuda_device_synchronize();
print_r($a->toArray());                 // [[2,4],[6,8]]

// CUDA graphs: capture once, replay cheaply
$stream = cuda_stream_create();
cuda_graph_begin_capture($stream);
cuda_kernel_launch($kernel, [$a, 4], [1], [256], $stream);
$graph = cuda_graph_end_capture();
cuda_graph_launch($graph, $stream);
cuda_stream_synchronize($stream);
```

## API overview

### CudaTensor (class)

| Group | Methods |
|---|---|
| Constructors | `fromArray`, `zeros`, `ones`, `full`, `rand` |
| Introspection | `shape`, `strides`, `dtype`, `ndim`, `size`, `nbytes`, `device`, `toArray`, `__toString` |
| Arithmetic | `add`, `sub`, `mul`, `div` (+ in-place `add_`, `sub_`, `mul_`, `div_`), `matmul` |
| Math | `relu`, `sigmoid`, `tanh`, `exp`, `log`, `sqrt`, `gelu`, `neg`, `softmax` |
| Reductions | `sum`, `mean`, `max`, `min` |
| Views | `reshape`, `transpose`, `slice`, `contiguous` |
| Constants | `CudaTensor::FP32`, `FP64`, `INT32`, `FP16`, `BF16`, `INT8` |

### Functions

| Group | Functions |
|---|---|
| Device | `cuda_device_count`, `cuda_device_properties`, `cuda_set_device`, `cuda_get_device`, `cuda_device_reset`, `cuda_device_synchronize`, `cuda_driver_version`, `cuda_runtime_version` |
| Memory | `cuda_malloc`, `cuda_free`, `cuda_memset`, `cuda_memcpy_host_to_device`, `cuda_memcpy_device_to_host`, `cuda_memcpy_device_to_device`, `cuda_pinned_alloc`, `cuda_unified_alloc`, `cuda_memory_get_info`, `cuda_measure_memory_bandwidth` |
| Pool | `cuda_memory_pool_init`, `cuda_memory_pool_allocate`, `cuda_memory_pool_free`, `cuda_memory_pool_stats`, `cuda_memory_pool_destroy` |
| Streams | `cuda_stream_create`, `cuda_stream_destroy`, `cuda_stream_synchronize`, `cuda_stream_query`, `cuda_stream_wait_event` |
| Events | `cuda_event_create`, `cuda_event_record_start`, `cuda_event_record_stop`, `cuda_event_elapsed_time`, `cuda_event_destroy` |
| Graphs | `cuda_graph_begin_capture`, `cuda_graph_end_capture`, `cuda_graph_launch`, `cuda_graph_destroy` |
| cuBLAS | `cuda_cublas_create`, `cuda_cublas_destroy`, `cuda_cublas_matrix_multiply`, `cuda_cublas_gemm`, `cuda_batch_gemm` |
| cuDNN | `cuda_cudnn_convolution_forward` (when built with cuDNN) |
| NVRTC | `cuda_kernel_compile`, `cuda_kernel_launch` (when built with NVRTC) |
| Errors | `cuda_get_last_error`, `cuda_get_error_string`, `cuda_get_error_name` |
| Profiling | `cuda_profiler_start`, `cuda_profiler_stop` |
| Legacy convenience | `cuda_matrix_multiply` (2-D PHP arrays) |

### ini settings

| Setting | Default | Meaning |
|---|---|---|
| `cuda.default_device` | `0` | Device selected at request start |
| `cuda.error_mode` | `warning` | `warning` (E_WARNING + false) or `exception` (throw `CudaException`) |
| `cuda.enable_cpu_fallback` | `1` | Allow CPU fallback for `cuda_matrix_multiply` on GPU-less hosts |
| `cuda.enable_memory_pool` | `0` | Reserved for the future pooled allocator default |

## Testing

```bash
cd plugin
php run-tests.php -q -d extension=$PWD/modules/cuda.so ../tests/
```

Tests skip gracefully when no GPU is present or when optional components
(cuDNN, NVRTC) are not compiled in.

## Contributing

Contributions welcome. The CI matrix builds PHP 8.1–8.4 × CUDA 11.8/12.x on
every PR; GPU tests run on a self-hosted runner.

## License

MIT License: see [LICENSE.MD](LICENSE.MD).

## Support

Issues and questions: GitHub issue tracker.
