#ifndef PHP_CUDA_H
#define PHP_CUDA_H

#include <cuda_runtime.h>
#include <cublas_v2.h>

extern zend_module_entry cuda_module_entry;
#define phpext_cuda_ptr &cuda_module_entry

#define PHP_CUDA_VERSION "0.2.0"
#define PHP_CUDA_EXTNAME "cuda"

/* -------------------------------------------------------------------------
 * Configuration
 * ------------------------------------------------------------------------- */
#define CUDA_TENSOR_MAX_DIMS 8

#define CUDA_ERROR_MODE_WARNING   0
#define CUDA_ERROR_MODE_EXCEPTION 1

#define CUDA_MEM_DEVICE  0
#define CUDA_MEM_PINNED  1
#define CUDA_MEM_UNIFIED 2

/* -------------------------------------------------------------------------
 * Module globals
 * ------------------------------------------------------------------------- */
ZEND_BEGIN_MODULE_GLOBALS(cuda)
    zend_bool enable_cpu_fallback;
    zend_bool enable_memory_pool;
    zend_long default_device;
    int current_device;
    int error_mode;
ZEND_END_MODULE_GLOBALS(cuda)

ZEND_EXTERN_MODULE_GLOBALS(cuda)

#define CUDA_G(v) ZEND_MODULE_GLOBALS_ACCESSOR(cuda, v)

/* -------------------------------------------------------------------------
 * Resource types (defined in cuda.c)
 * ------------------------------------------------------------------------- */
extern int le_cuda_memory;
extern int le_cuda_stream;
extern int le_cuda_event;
extern int le_cuda_graph;
extern int le_cublas_handle;
extern int le_cudnn_handle;
extern int le_cuda_kernel;
extern int le_memory_pool;

typedef struct _cuda_memory_resource {
    void *ptr;
    size_t size;
    int device_id;
    int kind; /* CUDA_MEM_* (3 = pool block) */
    void *pool; /* owning MemoryPool when kind == 3, else NULL */
} cuda_memory_resource;

typedef struct _cuda_stream_resource {
    cudaStream_t stream;
    int device_id;
} cuda_stream_resource;

typedef struct _cuda_event_resource {
    cudaEvent_t start;
    cudaEvent_t stop;
    int device_id;
} cuda_event_resource;

typedef struct _cuda_graph_resource {
    cudaGraphExec_t exec;
    int device_id;
} cuda_graph_resource;

typedef struct _cuda_cublas_resource {
    cublasHandle_t handle;
    int device_id;
} cuda_cublas_resource;

/* -------------------------------------------------------------------------
 * Exception class
 * ------------------------------------------------------------------------- */
extern zend_class_entry *cuda_exception_ce;

/* -------------------------------------------------------------------------
 * Error reporting.
 *
 * Default mode ("warning") raises E_WARNING and the caller returns false,
 * matching classic PHP extension behavior. "exception" mode throws
 * CudaException instead. Controlled via the cuda.error_mode ini setting.
 * ------------------------------------------------------------------------- */
void cuda_report_error(cudaError_t err, const char *file, int line);
void cuda_report_error_msg(const char *msg);

#define CUDA_CHECK_RET(err) \
    do { \
        cudaError_t _err = (err); \
        if (_err != cudaSuccess) { \
            cuda_report_error(_err, __FILE__, __LINE__); \
            RETURN_FALSE; \
        } \
    } while (0)

#define CUDA_CHECK_VOID(err) \
    do { \
        cudaError_t _err = (err); \
        if (_err != cudaSuccess) { \
            cuda_report_error(_err, __FILE__, __LINE__); \
            return; \
        } \
    } while (0)

/* -------------------------------------------------------------------------
 * Shared helpers (cuda.c)
 * ------------------------------------------------------------------------- */

/* Lazily switch to a device. Returns cudaSuccess or the CUDA error. */
cudaError_t cuda_use_device(int device_id);

/* Per-device cached cuBLAS handle (created lazily, destroyed at MSHUTDOWN). */
cublasHandle_t cuda_get_cublas_handle(int device_id);

/* Convert a PHP array of floats into a freshly emalloc'ed float buffer. */
float *php_cuda_array_to_floats(zval *arr, zend_long *count);
/* Build a PHP array from a float buffer. */
void php_cuda_floats_to_array(const float *buf, zend_long count, zval *out);

/* -------------------------------------------------------------------------
 * Module lifecycle
 * ------------------------------------------------------------------------- */
PHP_MINIT_FUNCTION(cuda);
PHP_MSHUTDOWN_FUNCTION(cuda);
PHP_RINIT_FUNCTION(cuda);
PHP_RSHUTDOWN_FUNCTION(cuda);
PHP_MINFO_FUNCTION(cuda);

/* -------------------------------------------------------------------------
 * Core: device management (cuda.c)
 * ------------------------------------------------------------------------- */
PHP_FUNCTION(cuda_device_count);
PHP_FUNCTION(cuda_device_properties);
PHP_FUNCTION(cuda_set_device);
PHP_FUNCTION(cuda_get_device);
PHP_FUNCTION(cuda_device_reset);
PHP_FUNCTION(cuda_device_synchronize);
PHP_FUNCTION(cuda_driver_version);
PHP_FUNCTION(cuda_runtime_version);

/* Core: memory management (cuda.c) */
PHP_FUNCTION(cuda_malloc);
PHP_FUNCTION(cuda_free);
PHP_FUNCTION(cuda_memset);
PHP_FUNCTION(cuda_memcpy_host_to_device);
PHP_FUNCTION(cuda_memcpy_device_to_host);
PHP_FUNCTION(cuda_memcpy_device_to_device);
PHP_FUNCTION(cuda_pinned_alloc);
PHP_FUNCTION(cuda_unified_alloc);
PHP_FUNCTION(cuda_memory_get_info);
PHP_FUNCTION(cuda_measure_memory_bandwidth);

/* Core: memory pool (cuda.c, backed by memory_pool.cu) */
PHP_FUNCTION(cuda_memory_pool_init);
PHP_FUNCTION(cuda_memory_pool_destroy);
PHP_FUNCTION(cuda_memory_pool_allocate);
PHP_FUNCTION(cuda_memory_pool_free);
PHP_FUNCTION(cuda_memory_pool_stats);

/* Core: compute + errors (cuda.c) */
PHP_FUNCTION(cuda_matrix_multiply);
PHP_FUNCTION(cuda_get_last_error);
PHP_FUNCTION(cuda_get_error_string);
PHP_FUNCTION(cuda_get_error_name);

/* Core: profiling (cuda.c) */
PHP_FUNCTION(cuda_profiler_start);
PHP_FUNCTION(cuda_profiler_stop);

/* -------------------------------------------------------------------------
 * Streams, events, graphs (streams.c)
 * ------------------------------------------------------------------------- */
PHP_FUNCTION(cuda_stream_create);
PHP_FUNCTION(cuda_stream_destroy);
PHP_FUNCTION(cuda_stream_synchronize);
PHP_FUNCTION(cuda_stream_query);
PHP_FUNCTION(cuda_stream_wait_event);
PHP_FUNCTION(cuda_event_create);
PHP_FUNCTION(cuda_event_destroy);
PHP_FUNCTION(cuda_event_record_start);
PHP_FUNCTION(cuda_event_record_stop);
PHP_FUNCTION(cuda_event_elapsed_time);
PHP_FUNCTION(cuda_graph_begin_capture);
PHP_FUNCTION(cuda_graph_end_capture);
PHP_FUNCTION(cuda_graph_launch);
PHP_FUNCTION(cuda_graph_destroy);

/* -------------------------------------------------------------------------
 * cuBLAS (cublas_ops.c)
 * ------------------------------------------------------------------------- */
PHP_FUNCTION(cuda_cublas_create);
PHP_FUNCTION(cuda_cublas_destroy);
PHP_FUNCTION(cuda_cublas_matrix_multiply);
PHP_FUNCTION(cuda_cublas_gemm);
PHP_FUNCTION(cuda_batch_gemm);

/* -------------------------------------------------------------------------
 * cuDNN (cudnn_ops.c, only when built with cuDNN)
 * ------------------------------------------------------------------------- */
#ifdef HAVE_CUDNN
PHP_FUNCTION(cuda_cudnn_convolution_forward);
#endif

/* -------------------------------------------------------------------------
 * NVRTC (nvrtc.c, only when built with NVRTC)
 * ------------------------------------------------------------------------- */
#ifdef HAVE_NVRTC
PHP_FUNCTION(cuda_kernel_compile);
PHP_FUNCTION(cuda_kernel_launch);
#endif

/* -------------------------------------------------------------------------
 * CudaTensor class (tensor.c)
 * ------------------------------------------------------------------------- */
extern zend_class_entry *cuda_tensor_ce;
void php_cuda_tensor_minit(void);
void php_cuda_tensor_mshutdown(void);

#if defined(ZTS) && defined(COMPILE_DL_CUDA)
ZEND_TSRMLS_CACHE_EXTERN()
#endif

#endif /* PHP_CUDA_H */
