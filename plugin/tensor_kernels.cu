#include "tensor_kernels.cuh"
#include "cuda_utils.cuh"

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math.h>

/* -------------------------------------------------------------------------
 * dtype traits: load/store through double (float for speed-critical paths is
 * a future optimization; correctness first).
 * ------------------------------------------------------------------------- */
template <int DT> struct dtype_traits;

template <> struct dtype_traits<CT_FP32> {
    typedef float type;
    static __device__ double load(const void *p, int64_t i) { return (double)((const type *)p)[i]; }
    static __device__ void store(void *p, int64_t i, double v) { ((type *)p)[i] = (type)v; }
};

template <> struct dtype_traits<CT_FP64> {
    typedef double type;
    static __device__ double load(const void *p, int64_t i) { return ((const type *)p)[i]; }
    static __device__ void store(void *p, int64_t i, double v) { ((type *)p)[i] = v; }
};

template <> struct dtype_traits<CT_INT32> {
    typedef int32_t type;
    static __device__ double load(const void *p, int64_t i) { return (double)((const type *)p)[i]; }
    static __device__ void store(void *p, int64_t i, double v) { ((type *)p)[i] = (type)llrint(v); }
};

template <> struct dtype_traits<CT_FP16> {
    typedef __half type;
    static __device__ double load(const void *p, int64_t i) { return (double)__half2float(((const type *)p)[i]); }
    static __device__ void store(void *p, int64_t i, double v) { ((type *)p)[i] = __float2half_rn((float)v); }
};

template <> struct dtype_traits<CT_BF16> {
    typedef __nv_bfloat16 type;
    static __device__ double load(const void *p, int64_t i) { return (double)__bfloat162float(((const type *)p)[i]); }
    static __device__ void store(void *p, int64_t i, double v) { ((type *)p)[i] = __float2bfloat16_rn((float)v); }
};

template <> struct dtype_traits<CT_INT8> {
    typedef int8_t type;
    static __device__ double load(const void *p, int64_t i) { return (double)((const type *)p)[i]; }
    static __device__ void store(void *p, int64_t i, double v) { ((type *)p)[i] = (type)llrint(v); }
};

/* -------------------------------------------------------------------------
 * Index decomposition: linear index -> element offset through strides.
 * ------------------------------------------------------------------------- */
__device__ inline int64_t ct_offset_of(int64_t linear, ct_dims shape, ct_dims strides, int ndim) {
    int64_t offset = 0;
    for (int d = ndim - 1; d >= 0; d--) {
        int64_t coord = linear % shape.v[d];
        linear /= shape.v[d];
        offset += coord * strides.v[d];
    }
    return offset;
}

/* -------------------------------------------------------------------------
 * Binary elementwise kernels
 * ------------------------------------------------------------------------- */
struct op_add { static __device__ double apply(double a, double b) { return a + b; } };
struct op_sub { static __device__ double apply(double a, double b) { return a - b; } };
struct op_mul { static __device__ double apply(double a, double b) { return a * b; } };
struct op_div { static __device__ double apply(double a, double b) { return a / b; } };

template <int DT, typename Op>
__global__ void binary_kernel(const void *a, ct_dims sa,
                              const void *b, ct_dims sb,
                              void *out, ct_dims shape, int ndim, int64_t total) {
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += stride) {
        int64_t off_a = ct_offset_of(idx, shape, sa, ndim);
        int64_t off_b = ct_offset_of(idx, shape, sb, ndim);
        double va = dtype_traits<DT>::load(a, off_a);
        double vb = dtype_traits<DT>::load(b, off_b);
        dtype_traits<DT>::store(out, idx, Op::apply(va, vb));
    }
}

/* -------------------------------------------------------------------------
 * Unary elementwise kernels
 * ------------------------------------------------------------------------- */
template <int DT>
__global__ void unary_kernel(int op, const void *in, ct_dims sin,
                             void *out, ct_dims shape, int ndim, int64_t total) {
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += stride) {
        int64_t off = ct_offset_of(idx, shape, sin, ndim);
        double v = dtype_traits<DT>::load(in, off);
        double r;
        switch (op) {
            case CT_UNARY_RELU:    r = v > 0.0 ? v : 0.0; break;
            case CT_UNARY_SIGMOID: r = 1.0 / (1.0 + exp(-v)); break;
            case CT_UNARY_TANH:    r = tanh(v); break;
            case CT_UNARY_EXP:     r = exp(v); break;
            case CT_UNARY_LOG:     r = log(v); break;
            case CT_UNARY_SQRT:    r = sqrt(v); break;
            case CT_UNARY_GELU:    r = 0.5 * v * (1.0 + tanh(0.7978845608028654 * (v + 0.044715 * v * v * v))); break;
            case CT_UNARY_NEG:     r = -v; break;
            default:               r = v; break;
        }
        dtype_traits<DT>::store(out, idx, r);
    }
}

/* -------------------------------------------------------------------------
 * Fill / gather / cast
 * ------------------------------------------------------------------------- */
template <int DT>
__global__ void fill_kernel(void *out, double value, int64_t total) {
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += stride) {
        dtype_traits<DT>::store(out, idx, value);
    }
}

template <int DT>
__global__ void gather_kernel(const void *in, ct_dims sin, void *out,
                              ct_dims shape, int ndim, int64_t total) {
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += stride) {
        int64_t off = ct_offset_of(idx, shape, sin, ndim);
        dtype_traits<DT>::store(out, idx, dtype_traits<DT>::load(in, off));
    }
}

template <int DT_IN, int DT_OUT>
__global__ void cast_kernel(const void *in, void *out, int64_t total) {
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += stride) {
        dtype_traits<DT_OUT>::store(out, idx, dtype_traits<DT_IN>::load(in, idx));
    }
}

/* -------------------------------------------------------------------------
 * Reduction (single pass, atomics into a device double; correctness first)
 * ------------------------------------------------------------------------- */
__device__ inline unsigned long long ct_ordered_bits(double v) {
    unsigned long long bits = __double_as_longlong(v);
    return (bits & 0x8000000000000000ULL) ? ~bits : (bits | 0x8000000000000000ULL);
}

__device__ inline double ct_from_ordered_bits(unsigned long long v) {
    unsigned long long bits = (v & 0x8000000000000000ULL) ? (v & 0x7FFFFFFFFFFFFFFFULL) : ~v;
    return __longlong_as_double(bits);
}

__device__ inline void ct_atomic_max_double(double *addr, double val) {
    atomicMax((unsigned long long *)addr, ct_ordered_bits(val));
}

__device__ inline void ct_atomic_min_double(double *addr, double val) {
    atomicMin((unsigned long long *)addr, ct_ordered_bits(val));
}

template <int DT>
__global__ void reduce_kernel(int op, const void *in, ct_dims sin,
                              ct_dims shape, int ndim, int64_t total, double *result) {
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += stride) {
        int64_t off = ct_offset_of(idx, shape, sin, ndim);
        double v = dtype_traits<DT>::load(in, off);
        switch (op) {
            case CT_REDUCE_SUM: atomicAdd(result, v); break;
            case CT_REDUCE_MAX: ct_atomic_max_double(result, v); break;
            case CT_REDUCE_MIN: ct_atomic_min_double(result, v); break;
        }
    }
}

/* -------------------------------------------------------------------------
 * Softmax over the last dimension (one block per row)
 * ------------------------------------------------------------------------- */
template <int DT>
__global__ void softmax_kernel(const void *in, void *out, int64_t rows, int64_t cols) {
    int64_t row = blockIdx.x;
    if (row >= rows) return;

    const int64_t base = row * cols;

    __shared__ double row_max;
    __shared__ double row_sum;

    if (threadIdx.x == 0) {
        row_max = -INFINITY;
        row_sum = 0.0;
    }
    __syncthreads();

    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) {
        double v = dtype_traits<DT>::load(in, base + c);
        ct_atomic_max_double(&row_max, v);
    }
    __syncthreads();

    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) {
        double v = dtype_traits<DT>::load(in, base + c);
        atomicAdd(&row_sum, exp(v - row_max));
    }
    __syncthreads();

    for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) {
        double v = dtype_traits<DT>::load(in, base + c);
        dtype_traits<DT>::store(out, base + c, exp(v - row_max) / row_sum);
    }
}

/* -------------------------------------------------------------------------
 * Launch helpers
 * ------------------------------------------------------------------------- */
static inline unsigned int ct_grid_for(int64_t total, int block) {
    int64_t blocks = (total + block - 1) / block;
    if (blocks > 65535) blocks = 65535;
    if (blocks < 1) blocks = 1;
    return (unsigned int)blocks;
}

#define CT_DISPATCH_DTYPE(dtype, TEMPLATE_CALL) \
    switch (dtype) { \
        case CT_FP32:  { TEMPLATE_CALL(CT_FP32);  break; } \
        case CT_FP64:  { TEMPLATE_CALL(CT_FP64);  break; } \
        case CT_INT32: { TEMPLATE_CALL(CT_INT32); break; } \
        case CT_FP16:  { TEMPLATE_CALL(CT_FP16);  break; } \
        case CT_BF16:  { TEMPLATE_CALL(CT_BF16);  break; } \
        case CT_INT8:  { TEMPLATE_CALL(CT_INT8);  break; } \
        default: return cudaErrorInvalidValue; \
    }

extern "C" size_t ct_dtype_size(int dtype) {
    switch (dtype) {
        case CT_FP32:  return 4;
        case CT_FP64:  return 8;
        case CT_INT32: return 4;
        case CT_FP16:  return 2;
        case CT_BF16:  return 2;
        case CT_INT8:  return 1;
        default:       return 0;
    }
}

extern "C" cudaError_t ct_elementwise_binary(
    int op,
    const void *a, ct_dims strides_a,
    const void *b, ct_dims strides_b,
    void *out,
    ct_dims shape, int ndim, int64_t total,
    int dtype, cudaStream_t stream
) {
    if (total <= 0) return cudaSuccess;
    int block = 256;
    unsigned int grid = ct_grid_for(total, block);

#define CT_BINARY_CALL(DT) \
    do { \
        switch (op) { \
            case CT_OP_ADD: binary_kernel<DT, op_add><<<grid, block, 0, stream>>>(a, strides_a, b, strides_b, out, shape, ndim, total); break; \
            case CT_OP_SUB: binary_kernel<DT, op_sub><<<grid, block, 0, stream>>>(a, strides_a, b, strides_b, out, shape, ndim, total); break; \
            case CT_OP_MUL: binary_kernel<DT, op_mul><<<grid, block, 0, stream>>>(a, strides_a, b, strides_b, out, shape, ndim, total); break; \
            case CT_OP_DIV: binary_kernel<DT, op_div><<<grid, block, 0, stream>>>(a, strides_a, b, strides_b, out, shape, ndim, total); break; \
            default: return cudaErrorInvalidValue; \
        } \
    } while (0)

    CT_DISPATCH_DTYPE(dtype, CT_BINARY_CALL);
#undef CT_BINARY_CALL
    return cudaGetLastError();
}

extern "C" cudaError_t ct_elementwise_unary(
    int op,
    const void *in, ct_dims strides_in,
    void *out,
    ct_dims shape, int ndim, int64_t total,
    int dtype, cudaStream_t stream
) {
    if (total <= 0) return cudaSuccess;
    if (op < 0 || op > CT_UNARY_NEG) return cudaErrorInvalidValue;
    int block = 256;
    unsigned int grid = ct_grid_for(total, block);

#define CT_UNARY_CALL(DT) unary_kernel<DT><<<grid, block, 0, stream>>>(op, in, strides_in, out, shape, ndim, total)
    CT_DISPATCH_DTYPE(dtype, CT_UNARY_CALL);
#undef CT_UNARY_CALL
    return cudaGetLastError();
}

extern "C" cudaError_t ct_fill(void *out, double value, int64_t total, int dtype, cudaStream_t stream) {
    if (total <= 0) return cudaSuccess;
    int block = 256;
    unsigned int grid = ct_grid_for(total, block);

#define CT_FILL_CALL(DT) fill_kernel<DT><<<grid, block, 0, stream>>>(out, value, total)
    CT_DISPATCH_DTYPE(dtype, CT_FILL_CALL);
#undef CT_FILL_CALL
    return cudaGetLastError();
}

extern "C" cudaError_t ct_gather(
    const void *in, ct_dims strides_in,
    void *out,
    ct_dims shape, int ndim, int64_t total,
    int dtype, cudaStream_t stream
) {
    if (total <= 0) return cudaSuccess;
    int block = 256;
    unsigned int grid = ct_grid_for(total, block);

#define CT_GATHER_CALL(DT) gather_kernel<DT><<<grid, block, 0, stream>>>(in, strides_in, out, shape, ndim, total)
    CT_DISPATCH_DTYPE(dtype, CT_GATHER_CALL);
#undef CT_GATHER_CALL
    return cudaGetLastError();
}

extern "C" cudaError_t ct_reduce(
    int op,
    const void *in, ct_dims strides_in,
    ct_dims shape, int ndim, int64_t total,
    int dtype, double *result, cudaStream_t stream
) {
    if (total <= 0 || !result) return cudaErrorInvalidValue;

    double init;
    switch (op) {
        case CT_REDUCE_SUM: init = 0.0; break;
        case CT_REDUCE_MAX: init = -INFINITY; break;
        case CT_REDUCE_MIN: init = INFINITY; break;
        default: return cudaErrorInvalidValue;
    }

    double *dev_result = nullptr;
    cudaError_t err = cudaMalloc((void **)&dev_result, sizeof(double));
    if (err != cudaSuccess) return err;

    err = cudaMemcpyAsync(dev_result, &init, sizeof(double), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) { cudaFree(dev_result); return err; }

    int block = 256;
    unsigned int grid = ct_grid_for(total, block);

#define CT_REDUCE_CALL(DT) reduce_kernel<DT><<<grid, block, 0, stream>>>(op, in, strides_in, shape, ndim, total, dev_result)
    CT_DISPATCH_DTYPE(dtype, CT_REDUCE_CALL);
#undef CT_REDUCE_CALL

    err = cudaGetLastError();
    if (err != cudaSuccess) { cudaFree(dev_result); return err; }

    err = cudaMemcpyAsync(result, dev_result, sizeof(double), cudaMemcpyDeviceToHost, stream);
    if (err == cudaSuccess) {
        err = cudaStreamSynchronize(stream);
    }
    cudaFree(dev_result);
    return err;
}

extern "C" cudaError_t ct_softmax(
    const void *in, void *out,
    int64_t rows, int64_t cols,
    int dtype, cudaStream_t stream
) {
    if (rows <= 0 || cols <= 0) return cudaErrorInvalidValue;
    int block = 256;
    unsigned int grid = (unsigned int)(rows > 65535 ? 65535 : rows);

#define CT_SOFTMAX_CALL(DT) softmax_kernel<DT><<<grid, block, 0, stream>>>(in, out, rows, cols)
    CT_DISPATCH_DTYPE(dtype, CT_SOFTMAX_CALL);
#undef CT_SOFTMAX_CALL
    return cudaGetLastError();
}

extern "C" cudaError_t ct_copy_cast(
    const void *in, void *out, int64_t total,
    int dtype_in, int dtype_out, cudaStream_t stream
) {
    if (total <= 0) return cudaSuccess;
    int block = 256;
    unsigned int grid = ct_grid_for(total, block);

#define CT_CAST_OUT(DT_IN) \
    switch (dtype_out) { \
        case CT_FP32:  cast_kernel<DT_IN, CT_FP32><<<grid, block, 0, stream>>>(in, out, total); break; \
        case CT_FP64:  cast_kernel<DT_IN, CT_FP64><<<grid, block, 0, stream>>>(in, out, total); break; \
        case CT_INT32: cast_kernel<DT_IN, CT_INT32><<<grid, block, 0, stream>>>(in, out, total); break; \
        case CT_FP16:  cast_kernel<DT_IN, CT_FP16><<<grid, block, 0, stream>>>(in, out, total); break; \
        case CT_BF16:  cast_kernel<DT_IN, CT_BF16><<<grid, block, 0, stream>>>(in, out, total); break; \
        case CT_INT8:  cast_kernel<DT_IN, CT_INT8><<<grid, block, 0, stream>>>(in, out, total); break; \
        default: return cudaErrorInvalidValue; \
    }

    switch (dtype_in) {
        case CT_FP32:  CT_CAST_OUT(CT_FP32);  break;
        case CT_FP64:  CT_CAST_OUT(CT_FP64);  break;
        case CT_INT32: CT_CAST_OUT(CT_INT32); break;
        case CT_FP16:  CT_CAST_OUT(CT_FP16);  break;
        case CT_BF16:  CT_CAST_OUT(CT_BF16);  break;
        case CT_INT8:  CT_CAST_OUT(CT_INT8);  break;
        default: return cudaErrorInvalidValue;
    }
#undef CT_CAST_OUT
    return cudaGetLastError();
}
