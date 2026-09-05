#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

// Internal utilities shared by the .cu translation units.
// PHP-facing error handling lives in php_cuda.h (CUDA_CHECK / CUDA_CHECK_RET);
// the macros below are for plain CUDA/C++ code and deliberately have
// different names to avoid collisions.

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#ifdef HAVE_CUDNN
#include <cudnn.h>
#endif
#include <string.h>
#include <stdio.h>

#if CUDART_VERSION < 11080
#error "CUDA 11.8 or higher is required"
#endif

// Return the error to the caller after logging (for extern "C" library-style
// functions that report status via cudaError_t).
#define CUDA_RT_CHECK(err) \
    do { \
        cudaError_t _e = (err); \
        if (_e != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s (%d): %s\n", \
                    __FILE__, __LINE__, cudaGetErrorName(_e), (int)_e, \
                    cudaGetErrorString(_e)); \
            return _e; \
        } \
    } while (0)

#define CUBLAS_RT_CHECK(err) \
    do { \
        cublasStatus_t _e = (err); \
        if (_e != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error in %s:%d: status %d\n", \
                    __FILE__, __LINE__, (int)_e); \
            return cudaErrorUnknown; \
        } \
    } while (0)

#ifdef HAVE_CUDNN
#define CUDNN_RT_CHECK(err) \
    do { \
        cudnnStatus_t _e = (err); \
        if (_e != CUDNN_STATUS_SUCCESS) { \
            fprintf(stderr, "cuDNN error in %s:%d: %s\n", \
                    __FILE__, __LINE__, cudnnGetErrorString(_e)); \
            return cudaErrorUnknown; \
        } \
    } while (0)
#endif

// Thread-local buffer for the last detailed error message.
static __thread char cuda_last_error_msg[1024];

inline const char *get_last_cuda_error_msg() {
    return cuda_last_error_msg;
}

inline void set_last_cuda_error_msg(const char *msg) {
    strncpy(cuda_last_error_msg, msg, sizeof(cuda_last_error_msg) - 1);
    cuda_last_error_msg[sizeof(cuda_last_error_msg) - 1] = '\0';
}

#endif // CUDA_UTILS_CUH
