#ifndef TENSOR_CORE_OPS_CUH
#define TENSOR_CORE_OPS_CUH

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include "cuda_utils.cuh"

// Tensor Core configuration
struct TensorCoreConfig {
    bool enabled;
    cublasComputeType_t compute_type;
    cudaDataType_t input_type;
    cudaDataType_t output_type;
    cublasGemmAlgo_t algo;
};

extern "C" {
    // Tensor Core operations
    cudaError_t cuda_tensorcore_matmul(
        cublasHandle_t handle,
        const void* A,
        const void* B,
        void* C,
        int m, int n, int k,
        TensorCoreConfig* config
    );
    
    // Mixed precision operations
    cudaError_t cuda_mixed_precision_matmul(
        cublasHandle_t handle,
        const half* A,
        const half* B,
        float* C,
        int m, int n, int k
    );
    
    // Auto-tuning
    cudaError_t cuda_tensorcore_autotune(
        int m, int n, int k,
        TensorCoreConfig* config
    );
}

#endif // TENSOR_CORE_OPS_CUH
