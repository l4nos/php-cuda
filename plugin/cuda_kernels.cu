#include "cuda_kernels.cuh"
#include "cuda_utils.cuh"

#define TILE 16

// Row-major SGEMM: C(m x p) = A(m x n) * B(n x p)
__global__ void matmul_kernel(const float *A, const float *B, float *C,
                              int m, int n, int p) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.0f;
    for (int t = 0; t < (n + TILE - 1) / TILE; t++) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] =
            (row < m && a_col < n) ? A[row * n + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] =
            (b_row < n && col < p) ? B[b_row * p + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < m && col < p) {
        C[row * p + col] = sum;
    }
}

__global__ void fill_float_kernel(float *ptr, float value, size_t count) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        ptr[idx] = value;
    }
}

extern "C" cudaError_t cuda_matrix_multiply_kernel_wrapper(
    const float *a, const float *b, float *c,
    int m, int n, int p,
    cudaStream_t stream
) {
    if (m <= 0 || n <= 0 || p <= 0) {
        return cudaErrorInvalidValue;
    }

    dim3 block(TILE, TILE);
    dim3 grid((p + TILE - 1) / TILE, (m + TILE - 1) / TILE);

    matmul_kernel<<<grid, block, 0, stream>>>(a, b, c, m, n, p);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return err;
    }
    if (stream == 0) {
        return cudaDeviceSynchronize();
    }
    return cudaSuccess;
}

extern "C" cudaError_t cuda_fill_float(float *ptr, float value, size_t count,
                                       cudaStream_t stream) {
    if (count == 0) {
        return cudaSuccess;
    }
    int blockSize = 256;
    size_t numBlocks = (count + blockSize - 1) / blockSize;
    fill_float_kernel<<<(unsigned int)numBlocks, blockSize, 0, stream>>>(ptr, value, count);
    return cudaGetLastError();
}
