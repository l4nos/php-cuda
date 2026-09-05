#ifndef TENSOR_KERNELS_CUH
#define TENSOR_KERNELS_CUH

/*
 * C-safe API for the CudaTensor compute kernels.
 * Included by both tensor.c (C) and tensor_kernels.cu (C++/NVCC).
 */

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

/* Data types supported by CudaTensor. */
#define CT_FP32 0
#define CT_FP64 1
#define CT_INT32 2
#define CT_FP16 3
#define CT_BF16 4
#define CT_INT8 5

/* Binary elementwise operations. */
#define CT_OP_ADD 0
#define CT_OP_SUB 1
#define CT_OP_MUL 2
#define CT_OP_DIV 3

/* Unary elementwise operations. */
#define CT_UNARY_RELU    0
#define CT_UNARY_SIGMOID 1
#define CT_UNARY_TANH    2
#define CT_UNARY_EXP     3
#define CT_UNARY_LOG     4
#define CT_UNARY_SQRT    5
#define CT_UNARY_GELU    6
#define CT_UNARY_NEG     7

/* Reduction operations. */
#define CT_REDUCE_SUM 0
#define CT_REDUCE_MAX 1
#define CT_REDUCE_MIN 2

#define CT_MAX_DIMS 8

/* Fixed-size shape/stride bundle passed to kernels by value. */
typedef struct _ct_dims {
    int64_t v[CT_MAX_DIMS];
} ct_dims;

#ifdef __cplusplus
extern "C" {
#endif

/* Elementwise binary op with broadcasting.
 * strides are in elements, already broadcast-adjusted (stride 0 where the
 * operand is broadcast). `out` is contiguous with `total` elements. */
cudaError_t ct_elementwise_binary(
    int op,
    const void *a, ct_dims strides_a,
    const void *b, ct_dims strides_b,
    void *out,
    ct_dims shape, int ndim, int64_t total,
    int dtype, cudaStream_t stream
);

/* Elementwise unary op. `in` may be strided; `out` is contiguous. */
cudaError_t ct_elementwise_unary(
    int op,
    const void *in, ct_dims strides_in,
    void *out,
    ct_dims shape, int ndim, int64_t total,
    int dtype, cudaStream_t stream
);

/* Fill a contiguous buffer with a constant. */
cudaError_t ct_fill(void *out, double value, int64_t total, int dtype, cudaStream_t stream);

/* Gather a (possibly strided) view into a contiguous buffer, same dtype. */
cudaError_t ct_gather(
    const void *in, ct_dims strides_in,
    void *out,
    ct_dims shape, int ndim, int64_t total,
    int dtype, cudaStream_t stream
);

/* Full reduction to a host-side double. */
cudaError_t ct_reduce(
    int op,
    const void *in, ct_dims strides_in,
    ct_dims shape, int ndim, int64_t total,
    int dtype, double *result, cudaStream_t stream
);

/* Softmax over the last dimension of a contiguous tensor.
 * rows * cols == total elements. */
cudaError_t ct_softmax(
    const void *in, void *out,
    int64_t rows, int64_t cols,
    int dtype, cudaStream_t stream
);

/* Copy with dtype conversion (both buffers contiguous). */
cudaError_t ct_copy_cast(
    const void *in, void *out, int64_t total,
    int dtype_in, int dtype_out, cudaStream_t stream
);

/* Number of bytes per element for a CT_* dtype, 0 if invalid. */
size_t ct_dtype_size(int dtype);

#ifdef __cplusplus
}
#endif

#endif // TENSOR_KERNELS_CUH
