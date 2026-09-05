#ifndef CUDA_KERNELS_CUH
#define CUDA_KERNELS_CUH

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Naive tiled SGEMM: C(m x p) = A(m x n) * B(n x p), all row-major.
 * Pass stream = 0 for the default stream. */
cudaError_t cuda_matrix_multiply_kernel_wrapper(
    const float *a, const float *b, float *c,
    int m, int n, int p,
    cudaStream_t stream
);

/* Fill a device buffer with a constant value. */
cudaError_t cuda_fill_float(float *ptr, float value, size_t count, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif // CUDA_KERNELS_CUH
