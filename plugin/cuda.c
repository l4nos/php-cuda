#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "php.h"
#include "ext/standard/info.h"
#include "php_ini.h"
#include "zend_exceptions.h"
#include "php_cuda.h"
#include "tensor.h"
#include "cuda_kernels.cuh"

ZEND_DECLARE_MODULE_GLOBALS(cuda)

/* -------------------------------------------------------------------------
 * Resource type IDs
 * ------------------------------------------------------------------------- */
int le_cuda_memory;
int le_cuda_stream;
int le_cuda_event;
int le_cuda_graph;
int le_cublas_handle;
int le_cudnn_handle;
int le_cuda_kernel;
int le_memory_pool;

zend_class_entry *cuda_exception_ce;

/* -------------------------------------------------------------------------
 * Opaque C++ pool API (memory_pool.cu)
 * ------------------------------------------------------------------------- */
typedef struct MemoryPool MemoryPool;
extern cudaError_t cuda_memory_pool_create(MemoryPool **pool, size_t initial_size, cudaStream_t stream);
extern cudaError_t cuda_memory_pool_allocate(MemoryPool *pool, size_t size, void **ptr);
extern cudaError_t cuda_memory_pool_free(MemoryPool *pool, void *ptr);
extern cudaError_t cuda_memory_pool_destroy(MemoryPool *pool);
extern cudaError_t cuda_memory_pool_get_stats(MemoryPool *pool, size_t *free_bytes, size_t *total_bytes);

/* CPU fallback (cpu_ops.cu) */
extern void cpu_matrix_multiply(const float *a, const float *b, float *c, int m, int n, int k);

/* Bandwidth measurement (memory_utils.cu) */
extern cudaError_t cuda_measure_memory_bandwidth(size_t size, float *bandwidth);

typedef struct _cuda_pool_shared {
    MemoryPool *pool;
    int device_id;
    uint32_t refs;      /* pool resource + outstanding blocks */
} cuda_pool_shared;

/* Destroy the shared pool state when the last reference goes away. Safe
 * against resource destruction order at request shutdown: blocks and the
 * pool resource each hold one reference; the last one out frees the pool. */
static void cuda_pool_shared_unref(cuda_pool_shared *shared) {
    if (!shared) return;
    if (--shared->refs == 0) {
        cudaSetDevice(shared->device_id);
        cuda_memory_pool_destroy(shared->pool);
        efree(shared);
    }
}

/* -------------------------------------------------------------------------
 * Resource destructors
 * ------------------------------------------------------------------------- */
static void cuda_memory_dtor(zend_resource *rsrc) {
    cuda_memory_resource *mem = (cuda_memory_resource *)rsrc->ptr;
    if (!mem) return;

    cudaSetDevice(mem->device_id);
    if (mem->kind == CUDA_MEM_PINNED) {
        cudaFreeHost(mem->ptr);
    } else if (mem->kind == 3 /* pool */) {
        /* Pool blocks are returned to their pool, not cudaFree'd. */
        cuda_pool_shared *shared = (cuda_pool_shared *)mem->pool;
        if (shared) {
            cuda_memory_pool_free(shared->pool, mem->ptr);
            cuda_pool_shared_unref(shared);
        }
    } else {
        cudaFree(mem->ptr);
    }
    efree(mem);
}

static void cuda_pool_dtor(zend_resource *rsrc) {
    cuda_pool_shared *shared = (cuda_pool_shared *)rsrc->ptr;
    cuda_pool_shared_unref(shared);
}

/* stream/event/graph/cublas dtors live in streams.c / cublas_ops.c */
extern void php_cuda_stream_dtor(zend_resource *rsrc);
extern void php_cuda_event_dtor(zend_resource *rsrc);
extern void php_cuda_graph_dtor(zend_resource *rsrc);
extern void php_cuda_cublas_dtor(zend_resource *rsrc);
#ifdef HAVE_NVRTC
extern void php_cuda_kernel_dtor(zend_resource *rsrc);
#endif

/* -------------------------------------------------------------------------
 * Error reporting
 * ------------------------------------------------------------------------- */
void cuda_report_error(cudaError_t err, const char *file, int line) {
    if (CUDA_G(error_mode) == CUDA_ERROR_MODE_EXCEPTION) {
        zend_throw_exception_ex(cuda_exception_ce, (zend_long)err,
            "CUDA error %s (%d): %s [%s:%d]",
            cudaGetErrorName(err), (int)err, cudaGetErrorString(err), file, line);
    } else {
        php_error_docref(NULL, E_WARNING, "CUDA error %s (%d): %s [%s:%d]",
            cudaGetErrorName(err), (int)err, cudaGetErrorString(err), file, line);
    }
}

void cuda_report_error_msg(const char *msg) {
    if (CUDA_G(error_mode) == CUDA_ERROR_MODE_EXCEPTION) {
        zend_throw_exception(cuda_exception_ce, msg, 0);
    } else {
        php_error_docref(NULL, E_WARNING, "%s", msg);
    }
}

/* -------------------------------------------------------------------------
 * Device helpers
 * ------------------------------------------------------------------------- */
cudaError_t cuda_use_device(int device_id) {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) return err;
    if (device_id < 0 || device_id >= count) return cudaErrorInvalidDevice;

    int current = -1;
    err = cudaGetDevice(&current);
    if (err != cudaSuccess) return err;
    if (current != device_id) {
        return cudaSetDevice(device_id);
    }
    return cudaSuccess;
}

#define CUDA_MAX_HANDLE_DEVICES 64
static cublasHandle_t php_cublas_handles[CUDA_MAX_HANDLE_DEVICES] = {NULL};

cublasHandle_t cuda_get_cublas_handle(int device_id) {
    if (device_id < 0 || device_id >= CUDA_MAX_HANDLE_DEVICES) return NULL;
    if (!php_cublas_handles[device_id]) {
        if (cuda_use_device(device_id) != cudaSuccess) return NULL;
        if (cublasCreate(&php_cublas_handles[device_id]) != CUBLAS_STATUS_SUCCESS) {
            return NULL;
        }
    }
    return php_cublas_handles[device_id];
}

static void cuda_destroy_cublas_handles(void) {
    for (int i = 0; i < CUDA_MAX_HANDLE_DEVICES; i++) {
        if (php_cublas_handles[i]) {
            cudaSetDevice(i);
            cublasDestroy(php_cublas_handles[i]);
            php_cublas_handles[i] = NULL;
        }
    }
}

/* -------------------------------------------------------------------------
 * PHP array <-> float buffer helpers
 * ------------------------------------------------------------------------- */
float *php_cuda_array_to_floats(zval *arr, zend_long *count) {
    HashTable *ht = Z_ARRVAL_P(arr);
    zend_long n = zend_hash_num_elements(ht);
    float *buf = emalloc((size_t)(n ? n : 1) * sizeof(float));

    zend_long i = 0;
    zval *zv;
    ZEND_HASH_FOREACH_VAL(ht, zv) {
        buf[i++] = (float)zval_get_double(zv);
    } ZEND_HASH_FOREACH_END();

    *count = n;
    return buf;
}

void php_cuda_floats_to_array(const float *buf, zend_long count, zval *out) {
    array_init_size(out, (uint32_t)count);
    for (zend_long i = 0; i < count; i++) {
        add_next_index_double(out, (double)buf[i]);
    }
}

/* -------------------------------------------------------------------------
 * ini settings
 * ------------------------------------------------------------------------- */
static PHP_INI_MH(OnUpdateErrorMode) {
    if (new_value && ZSTR_VAL(new_value) && strcmp(ZSTR_VAL(new_value), "exception") == 0) {
        CUDA_G(error_mode) = CUDA_ERROR_MODE_EXCEPTION;
    } else {
        CUDA_G(error_mode) = CUDA_ERROR_MODE_WARNING;
    }
    return SUCCESS;
}

PHP_INI_BEGIN()
    STD_PHP_INI_ENTRY("cuda.default_device", "0", PHP_INI_ALL, OnUpdateLong, default_device, zend_cuda_globals, cuda_globals)
    STD_PHP_INI_ENTRY("cuda.enable_cpu_fallback", "1", PHP_INI_ALL, OnUpdateBool, enable_cpu_fallback, zend_cuda_globals, cuda_globals)
    STD_PHP_INI_ENTRY("cuda.enable_memory_pool", "0", PHP_INI_ALL, OnUpdateBool, enable_memory_pool, zend_cuda_globals, cuda_globals)
    PHP_INI_ENTRY("cuda.error_mode", "warning", PHP_INI_ALL, OnUpdateErrorMode)
PHP_INI_END()

/* -------------------------------------------------------------------------
 * Module lifecycle
 * ------------------------------------------------------------------------- */
PHP_MINIT_FUNCTION(cuda) {
    REGISTER_INI_ENTRIES();

    zend_class_entry ce;
    INIT_CLASS_ENTRY(ce, "CudaException", NULL);
    cuda_exception_ce = zend_register_internal_class_ex(&ce, zend_ce_exception);

    le_cuda_memory = zend_register_list_destructors_ex(cuda_memory_dtor, NULL, "CUDA Memory", module_number);
    le_cuda_stream = zend_register_list_destructors_ex(php_cuda_stream_dtor, NULL, "CUDA Stream", module_number);
    le_cuda_event = zend_register_list_destructors_ex(php_cuda_event_dtor, NULL, "CUDA Event", module_number);
    le_cuda_graph = zend_register_list_destructors_ex(php_cuda_graph_dtor, NULL, "CUDA Graph", module_number);
    le_cublas_handle = zend_register_list_destructors_ex(php_cuda_cublas_dtor, NULL, "cuBLAS Handle", module_number);
    le_memory_pool = zend_register_list_destructors_ex(cuda_pool_dtor, NULL, "CUDA Memory Pool", module_number);
#ifdef HAVE_NVRTC
    le_cuda_kernel = zend_register_list_destructors_ex(php_cuda_kernel_dtor, NULL, "CUDA Kernel", module_number);
#endif

    php_cuda_tensor_minit();

    /* Deliberately no cudaSetDevice() here: initialization is lazy so the
     * extension loads on GPU-less hosts (web servers, CI builders). */
    return SUCCESS;
}

PHP_MSHUTDOWN_FUNCTION(cuda) {
    UNREGISTER_INI_ENTRIES();
    cuda_destroy_cublas_handles();
    php_cuda_tensor_mshutdown();
    /* No implicit cudaDeviceReset(): it would tear down the CUDA context for
     * the entire process, which is hostile under php-fpm and persistent
     * workers. Use cuda_device_reset() explicitly if you need it. */
    return SUCCESS;
}

PHP_RINIT_FUNCTION(cuda) {
#if defined(ZTS) && defined(COMPILE_DL_CUDA)
    ZEND_TSRMLS_CACHE_UPDATE();
#endif
    CUDA_G(current_device) = (int)CUDA_G(default_device);
    if (CUDA_G(error_mode) != CUDA_ERROR_MODE_EXCEPTION) {
        CUDA_G(error_mode) = CUDA_ERROR_MODE_WARNING;
    }
    return SUCCESS;
}

PHP_RSHUTDOWN_FUNCTION(cuda) {
    return SUCCESS;
}

PHP_MINFO_FUNCTION(cuda) {
    php_info_print_table_start();
    php_info_print_table_header(2, "CUDA Support", "enabled");
    php_info_print_table_row(2, "Extension Version", PHP_CUDA_VERSION);

    int driver_version = 0;
    if (cudaDriverGetVersion(&driver_version) == cudaSuccess && driver_version > 0) {
        char buf[32];
        snprintf(buf, sizeof(buf), "%d.%d", driver_version / 1000, (driver_version % 100) / 10);
        php_info_print_table_row(2, "CUDA Driver Version", buf);
    }

    int runtime_version = 0;
    if (cudaRuntimeGetVersion(&runtime_version) == cudaSuccess) {
        char buf[32];
        snprintf(buf, sizeof(buf), "%d.%d", runtime_version / 1000, (runtime_version % 100) / 10);
        php_info_print_table_row(2, "CUDA Runtime Version", buf);
    }

#ifdef HAVE_CUDNN
    php_info_print_table_row(2, "cuDNN", "enabled");
#else
    php_info_print_table_row(2, "cuDNN", "disabled");
#endif
#ifdef HAVE_NVRTC
    php_info_print_table_row(2, "NVRTC", "enabled");
#else
    php_info_print_table_row(2, "NVRTC", "disabled");
#endif
#ifdef HAVE_NVTX
    php_info_print_table_row(2, "NVTX", "enabled");
#else
    php_info_print_table_row(2, "NVTX", "disabled");
#endif

    int count = 0;
    if (cudaGetDeviceCount(&count) == cudaSuccess) {
        char buf[16];
        snprintf(buf, sizeof(buf), "%d", count);
        php_info_print_table_row(2, "CUDA Devices", buf);
    }

    php_info_print_table_end();
    DISPLAY_INI_ENTRIES();
}

/* -------------------------------------------------------------------------
 * Device management
 * ------------------------------------------------------------------------- */
PHP_FUNCTION(cuda_device_count) {
    ZEND_PARSE_PARAMETERS_NONE();
    int count = 0;
    CUDA_CHECK_RET(cudaGetDeviceCount(&count));
    RETURN_LONG(count);
}

PHP_FUNCTION(cuda_device_properties) {
    zend_long device_id;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_LONG(device_id)
    ZEND_PARSE_PARAMETERS_END();

    struct cudaDeviceProp props;
    CUDA_CHECK_RET(cudaGetDeviceProperties(&props, (int)device_id));

    array_init(return_value);
    add_assoc_string(return_value, "name", props.name);
    add_assoc_long(return_value, "totalGlobalMem", (zend_long)props.totalGlobalMem);
    add_assoc_long(return_value, "sharedMemPerBlock", (zend_long)props.sharedMemPerBlock);
    add_assoc_long(return_value, "regsPerBlock", props.regsPerBlock);
    add_assoc_long(return_value, "warpSize", props.warpSize);
    add_assoc_long(return_value, "maxThreadsPerBlock", props.maxThreadsPerBlock);
    add_assoc_long(return_value, "multiProcessorCount", props.multiProcessorCount);
    add_assoc_long(return_value, "clockRate", props.clockRate);
    add_assoc_long(return_value, "memoryClockRate", props.memoryClockRate);
    add_assoc_long(return_value, "memoryBusWidth", props.memoryBusWidth);
    add_assoc_long(return_value, "l2CacheSize", props.l2CacheSize);
    add_assoc_long(return_value, "computeCapabilityMajor", props.major);
    add_assoc_long(return_value, "computeCapabilityMinor", props.minor);
    add_assoc_bool(return_value, "unifiedAddressing", props.unifiedAddressing);
    add_assoc_bool(return_value, "managedMemory", props.managedMemory);
    add_assoc_bool(return_value, "concurrentKernels", props.concurrentKernels);

    zval max_threads_dim, max_grid_size;
    array_init(&max_threads_dim);
    array_init(&max_grid_size);
    for (int i = 0; i < 3; i++) {
        add_next_index_long(&max_threads_dim, props.maxThreadsDim[i]);
        add_next_index_long(&max_grid_size, props.maxGridSize[i]);
    }
    add_assoc_zval(return_value, "maxThreadsDim", &max_threads_dim);
    add_assoc_zval(return_value, "maxGridSize", &max_grid_size);
}

PHP_FUNCTION(cuda_set_device) {
    zend_long device_id;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_LONG(device_id)
    ZEND_PARSE_PARAMETERS_END();

    CUDA_CHECK_RET(cuda_use_device((int)device_id));
    CUDA_G(current_device) = (int)device_id;
    RETURN_TRUE;
}

PHP_FUNCTION(cuda_get_device) {
    ZEND_PARSE_PARAMETERS_NONE();
    RETURN_LONG(CUDA_G(current_device));
}

PHP_FUNCTION(cuda_device_reset) {
    ZEND_PARSE_PARAMETERS_NONE();
    cuda_destroy_cublas_handles();
    CUDA_CHECK_RET(cudaDeviceReset());
    RETURN_TRUE;
}

PHP_FUNCTION(cuda_device_synchronize) {
    ZEND_PARSE_PARAMETERS_NONE();
    CUDA_CHECK_RET(cuda_use_device(CUDA_G(current_device)));
    CUDA_CHECK_RET(cudaDeviceSynchronize());
    RETURN_TRUE;
}

PHP_FUNCTION(cuda_driver_version) {
    ZEND_PARSE_PARAMETERS_NONE();
    int v = 0;
    CUDA_CHECK_RET(cudaDriverGetVersion(&v));
    char buf[32];
    snprintf(buf, sizeof(buf), "%d.%d", v / 1000, (v % 100) / 10);
    RETURN_STRING(buf);
}

PHP_FUNCTION(cuda_runtime_version) {
    ZEND_PARSE_PARAMETERS_NONE();
    int v = 0;
    CUDA_CHECK_RET(cudaRuntimeGetVersion(&v));
    char buf[32];
    snprintf(buf, sizeof(buf), "%d.%d", v / 1000, (v % 100) / 10);
    RETURN_STRING(buf);
}

/* -------------------------------------------------------------------------
 * Memory management
 * ------------------------------------------------------------------------- */
static cuda_memory_resource *cuda_mem_register(void *ptr, size_t size, int kind) {
    cuda_memory_resource *mem = emalloc(sizeof(cuda_memory_resource));
    mem->ptr = ptr;
    mem->size = size;
    mem->kind = kind;
    mem->pool = NULL;
    cudaGetDevice(&mem->device_id);
    return mem;
}

PHP_FUNCTION(cuda_malloc) {
    zend_long size;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_LONG(size)
    ZEND_PARSE_PARAMETERS_END();

    if (size <= 0) {
        cuda_report_error_msg("cuda_malloc: size must be positive");
        RETURN_FALSE;
    }

    CUDA_CHECK_RET(cuda_use_device(CUDA_G(current_device)));

    void *ptr = NULL;
    CUDA_CHECK_RET(cudaMalloc(&ptr, (size_t)size));
    RETURN_RES(zend_register_resource(cuda_mem_register(ptr, (size_t)size, CUDA_MEM_DEVICE), le_cuda_memory));
}

PHP_FUNCTION(cuda_pinned_alloc) {
    zend_long size;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_LONG(size)
    ZEND_PARSE_PARAMETERS_END();

    if (size <= 0) {
        cuda_report_error_msg("cuda_pinned_alloc: size must be positive");
        RETURN_FALSE;
    }

    void *ptr = NULL;
    CUDA_CHECK_RET(cudaHostAlloc(&ptr, (size_t)size, cudaHostAllocDefault));
    RETURN_RES(zend_register_resource(cuda_mem_register(ptr, (size_t)size, CUDA_MEM_PINNED), le_cuda_memory));
}

PHP_FUNCTION(cuda_unified_alloc) {
    zend_long size;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_LONG(size)
    ZEND_PARSE_PARAMETERS_END();

    if (size <= 0) {
        cuda_report_error_msg("cuda_unified_alloc: size must be positive");
        RETURN_FALSE;
    }

    CUDA_CHECK_RET(cuda_use_device(CUDA_G(current_device)));

    void *ptr = NULL;
    CUDA_CHECK_RET(cudaMallocManaged(&ptr, (size_t)size, cudaMemAttachGlobal));
    RETURN_RES(zend_register_resource(cuda_mem_register(ptr, (size_t)size, CUDA_MEM_UNIFIED), le_cuda_memory));
}

PHP_FUNCTION(cuda_free) {
    zval *res;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_RESOURCE(res)
    ZEND_PARSE_PARAMETERS_END();

    if (zend_fetch_resource(Z_RES_P(res), "CUDA Memory", le_cuda_memory) == NULL) {
        RETURN_FALSE;
    }
    /* The resource destructor owns the actual cudaFree/cudaFreeHost call.
     * Closing the list entry is all we do here (no double free). */
    zend_list_close(Z_RES_P(res));
    RETURN_TRUE;
}

PHP_FUNCTION(cuda_memset) {
    zval *res;
    zend_long value;
    zend_long size = -1;
    ZEND_PARSE_PARAMETERS_START(2, 3)
        Z_PARAM_RESOURCE(res)
        Z_PARAM_LONG(value)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(size)
    ZEND_PARSE_PARAMETERS_END();

    cuda_memory_resource *mem = (cuda_memory_resource *)zend_fetch_resource(
        Z_RES_P(res), "CUDA Memory", le_cuda_memory);
    if (!mem) RETURN_FALSE;

    size_t n = size < 0 ? mem->size : (size_t)size;
    if (n > mem->size) {
        cuda_report_error_msg("cuda_memset: size exceeds allocation");
        RETURN_FALSE;
    }

    CUDA_CHECK_RET(cuda_use_device(mem->device_id));
    CUDA_CHECK_RET(cudaMemset(mem->ptr, (int)value, n));
    RETURN_TRUE;
}

PHP_FUNCTION(cuda_memcpy_host_to_device) {
    zval *res;
    zend_string *data;
    zend_long offset = 0;
    ZEND_PARSE_PARAMETERS_START(2, 3)
        Z_PARAM_RESOURCE(res)
        Z_PARAM_STR(data)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(offset)
    ZEND_PARSE_PARAMETERS_END();

    cuda_memory_resource *mem = (cuda_memory_resource *)zend_fetch_resource(
        Z_RES_P(res), "CUDA Memory", le_cuda_memory);
    if (!mem) RETURN_FALSE;

    if (offset < 0 || (size_t)offset + ZSTR_LEN(data) > mem->size) {
        cuda_report_error_msg("cuda_memcpy_host_to_device: data exceeds allocation");
        RETURN_FALSE;
    }

    CUDA_CHECK_RET(cuda_use_device(mem->device_id));
    CUDA_CHECK_RET(cudaMemcpy((char *)mem->ptr + offset, ZSTR_VAL(data), ZSTR_LEN(data), cudaMemcpyHostToDevice));
    RETURN_TRUE;
}

PHP_FUNCTION(cuda_memcpy_device_to_host) {
    zval *res;
    zend_long size = -1;
    zend_long offset = 0;
    ZEND_PARSE_PARAMETERS_START(1, 3)
        Z_PARAM_RESOURCE(res)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(size)
        Z_PARAM_LONG(offset)
    ZEND_PARSE_PARAMETERS_END();

    cuda_memory_resource *mem = (cuda_memory_resource *)zend_fetch_resource(
        Z_RES_P(res), "CUDA Memory", le_cuda_memory);
    if (!mem) RETURN_FALSE;

    size_t n = size < 0 ? mem->size : (size_t)size;
    if (offset < 0 || (size_t)offset + n > mem->size) {
        cuda_report_error_msg("cuda_memcpy_device_to_host: range exceeds allocation");
        RETURN_FALSE;
    }

    zend_string *out = zend_string_alloc(n, 0);
    CUDA_CHECK_RET(cuda_use_device(mem->device_id));
    cudaError_t err = cudaMemcpy(ZSTR_VAL(out), (char *)mem->ptr + offset, n, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        zend_string_release(out);
        CUDA_CHECK_RET(err);
    }
    ZSTR_VAL(out)[n] = '\0';
    RETURN_STR(out);
}

PHP_FUNCTION(cuda_memcpy_device_to_device) {
    zval *dst_res, *src_res;
    zend_long size = -1;
    ZEND_PARSE_PARAMETERS_START(2, 3)
        Z_PARAM_RESOURCE(dst_res)
        Z_PARAM_RESOURCE(src_res)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(size)
    ZEND_PARSE_PARAMETERS_END();

    cuda_memory_resource *dst = (cuda_memory_resource *)zend_fetch_resource(
        Z_RES_P(dst_res), "CUDA Memory", le_cuda_memory);
    cuda_memory_resource *src = (cuda_memory_resource *)zend_fetch_resource(
        Z_RES_P(src_res), "CUDA Memory", le_cuda_memory);
    if (!dst || !src) RETURN_FALSE;

    size_t n = size < 0 ? (dst->size < src->size ? dst->size : src->size) : (size_t)size;
    if (n > dst->size || n > src->size) {
        cuda_report_error_msg("cuda_memcpy_device_to_device: size exceeds allocation");
        RETURN_FALSE;
    }

    CUDA_CHECK_RET(cuda_use_device(dst->device_id));
    CUDA_CHECK_RET(cudaMemcpy(dst->ptr, src->ptr, n, cudaMemcpyDeviceToDevice));
    RETURN_TRUE;
}

PHP_FUNCTION(cuda_memory_get_info) {
    ZEND_PARSE_PARAMETERS_NONE();
    CUDA_CHECK_RET(cuda_use_device(CUDA_G(current_device)));

    size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK_RET(cudaMemGetInfo(&free_bytes, &total_bytes));

    array_init(return_value);
    add_assoc_long(return_value, "free", (zend_long)free_bytes);
    add_assoc_long(return_value, "total", (zend_long)total_bytes);
    add_assoc_long(return_value, "used", (zend_long)(total_bytes - free_bytes));
}

PHP_FUNCTION(cuda_measure_memory_bandwidth) {
    zend_long size;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_LONG(size)
    ZEND_PARSE_PARAMETERS_END();

    if (size <= 0) {
        cuda_report_error_msg("cuda_measure_memory_bandwidth: size must be positive");
        RETURN_FALSE;
    }

    CUDA_CHECK_RET(cuda_use_device(CUDA_G(current_device)));

    float bandwidth = 0.0f;
    CUDA_CHECK_RET(cuda_measure_memory_bandwidth((size_t)size, &bandwidth));
    RETURN_DOUBLE((double)bandwidth);
}

/* -------------------------------------------------------------------------
 * Memory pool
 * ------------------------------------------------------------------------- */
PHP_FUNCTION(cuda_memory_pool_init) {
    zend_long initial_size;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_LONG(initial_size)
    ZEND_PARSE_PARAMETERS_END();

    if (initial_size <= 0) {
        cuda_report_error_msg("cuda_memory_pool_init: initial size must be positive");
        RETURN_FALSE;
    }

    CUDA_CHECK_RET(cuda_use_device(CUDA_G(current_device)));

    cuda_pool_shared *shared = emalloc(sizeof(cuda_pool_shared));
    shared->device_id = CUDA_G(current_device);
    shared->refs = 1;
    CUDA_CHECK_RET(cuda_memory_pool_create(&shared->pool, (size_t)initial_size, 0));

    RETURN_RES(zend_register_resource(shared, le_memory_pool));
}

PHP_FUNCTION(cuda_memory_pool_destroy) {
    zval *res;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_RESOURCE(res)
    ZEND_PARSE_PARAMETERS_END();

    if (zend_fetch_resource(Z_RES_P(res), "CUDA Memory Pool", le_memory_pool) == NULL) {
        RETURN_FALSE;
    }
    zend_list_close(Z_RES_P(res));
    RETURN_TRUE;
}

PHP_FUNCTION(cuda_memory_pool_allocate) {
    zval *res;
    zend_long size;
    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_RESOURCE(res)
        Z_PARAM_LONG(size)
    ZEND_PARSE_PARAMETERS_END();

    cuda_pool_shared *shared = (cuda_pool_shared *)zend_fetch_resource(
        Z_RES_P(res), "CUDA Memory Pool", le_memory_pool);
    if (!shared) RETURN_FALSE;

    if (size <= 0) {
        cuda_report_error_msg("cuda_memory_pool_allocate: size must be positive");
        RETURN_FALSE;
    }

    CUDA_CHECK_RET(cuda_use_device(shared->device_id));

    void *ptr = NULL;
    CUDA_CHECK_RET(cuda_memory_pool_allocate(shared->pool, (size_t)size, &ptr));

    cuda_memory_resource *mem = cuda_mem_register(ptr, (size_t)size, 3 /* pool */);
    mem->pool = shared;
    shared->refs++;
    RETURN_RES(zend_register_resource(mem, le_cuda_memory));
}

PHP_FUNCTION(cuda_memory_pool_free) {
    zval *pool_res, *mem_res;
    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_RESOURCE(pool_res)
        Z_PARAM_RESOURCE(mem_res)
    ZEND_PARSE_PARAMETERS_END();

    if (zend_fetch_resource(Z_RES_P(pool_res), "CUDA Memory Pool", le_memory_pool) == NULL) {
        RETURN_FALSE;
    }
    if (zend_fetch_resource(Z_RES_P(mem_res), "CUDA Memory", le_cuda_memory) == NULL) {
        RETURN_FALSE;
    }
    /* Closing the memory resource returns the block to the pool via its
     * destructor. */
    zend_list_close(Z_RES_P(mem_res));
    RETURN_TRUE;
}

PHP_FUNCTION(cuda_memory_pool_stats) {
    zval *res;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_RESOURCE(res)
    ZEND_PARSE_PARAMETERS_END();

    cuda_pool_shared *shared = (cuda_pool_shared *)zend_fetch_resource(
        Z_RES_P(res), "CUDA Memory Pool", le_memory_pool);
    if (!shared) RETURN_FALSE;

    size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK_RET(cuda_memory_pool_get_stats(shared->pool, &free_bytes, &total_bytes));

    array_init(return_value);
    add_assoc_long(return_value, "total_size", (zend_long)total_bytes);
    add_assoc_long(return_value, "free_size", (zend_long)free_bytes);
    add_assoc_long(return_value, "used_size", (zend_long)(total_bytes - free_bytes));
}

/* -------------------------------------------------------------------------
 * Matrix multiply (2-D PHP arrays; convenience API over the internal kernel)
 * ------------------------------------------------------------------------- */
PHP_FUNCTION(cuda_matrix_multiply) {
    zval *matrix_a, *matrix_b, *result;
    ZEND_PARSE_PARAMETERS_START(3, 3)
        Z_PARAM_ARRAY(matrix_a)
        Z_PARAM_ARRAY(matrix_b)
        Z_PARAM_ZVAL(result)
    ZEND_PARSE_PARAMETERS_END();

    HashTable *ht_a = Z_ARRVAL_P(matrix_a);
    HashTable *ht_b = Z_ARRVAL_P(matrix_b);

    zend_long m = zend_hash_num_elements(ht_a);
    if (m == 0) {
        cuda_report_error_msg("cuda_matrix_multiply: matrix A is empty");
        RETURN_FALSE;
    }

    zval *row0 = zend_hash_index_find(ht_a, 0);
    if (!row0 || Z_TYPE_P(row0) != IS_ARRAY) {
        cuda_report_error_msg("cuda_matrix_multiply: matrix A must be a 2-D array");
        RETURN_FALSE;
    }
    zend_long n = zend_hash_num_elements(Z_ARRVAL_P(row0)); /* cols of A */

    zend_long n_b = zend_hash_num_elements(ht_b);           /* rows of B */
    if (n == 0 || n_b == 0) {
        cuda_report_error_msg("cuda_matrix_multiply: matrices have zero dimensions");
        RETURN_FALSE;
    }
    if (n != n_b) {
        cuda_report_error_msg("cuda_matrix_multiply: inner dimensions do not match (cols(A) != rows(B))");
        RETURN_FALSE;
    }

    zval *b_row0 = zend_hash_index_find(ht_b, 0);
    if (!b_row0 || Z_TYPE_P(b_row0) != IS_ARRAY) {
        cuda_report_error_msg("cuda_matrix_multiply: matrix B must be a 2-D array");
        RETURN_FALSE;
    }
    zend_long p = zend_hash_num_elements(Z_ARRVAL_P(b_row0)); /* cols of B */
    if (p == 0) {
        cuda_report_error_msg("cuda_matrix_multiply: matrix B has zero columns");
        RETURN_FALSE;
    }

    /* Flatten with strict rectangular validation. Counters are used instead
     * of hash keys so sparse/non-sequential keys cannot overflow buffers. */
    float *host_a = emalloc((size_t)m * n * sizeof(float));
    float *host_b = emalloc((size_t)n * p * sizeof(float));
    float *host_c = emalloc((size_t)m * p * sizeof(float));

    zval *row, *val;
    zend_bool valid = 1;
    zend_long ri, ci;

    ri = 0;
    ZEND_HASH_FOREACH_VAL(ht_a, row) {
        if (Z_TYPE_P(row) != IS_ARRAY || zend_hash_num_elements(Z_ARRVAL_P(row)) != (uint32_t)n) {
            valid = 0;
            break;
        }
        ci = 0;
        ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(row), val) {
            host_a[ri * n + ci] = (float)zval_get_double(val);
            ci++;
        } ZEND_HASH_FOREACH_END();
        ri++;
    } ZEND_HASH_FOREACH_END();

    if (valid) {
        ri = 0;
        ZEND_HASH_FOREACH_VAL(ht_b, row) {
            if (Z_TYPE_P(row) != IS_ARRAY || zend_hash_num_elements(Z_ARRVAL_P(row)) != (uint32_t)p) {
                valid = 0;
                break;
            }
            ci = 0;
            ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(row), val) {
                host_b[ri * p + ci] = (float)zval_get_double(val);
                ci++;
            } ZEND_HASH_FOREACH_END();
            ri++;
        } ZEND_HASH_FOREACH_END();
    }

    if (!valid) {
        efree(host_a);
        efree(host_b);
        efree(host_c);
        cuda_report_error_msg("cuda_matrix_multiply: matrices must be rectangular 2-D arrays");
        RETURN_FALSE;
    }

    int device_count = 0;
    cudaGetDeviceCount(&device_count);

    cudaError_t err = cudaSuccess;

    if (device_count > 0) {
        /* GPU path */
        err = cuda_use_device(CUDA_G(current_device));
        float *dev_a = NULL, *dev_b = NULL, *dev_c = NULL;

        if (err == cudaSuccess) err = cudaMalloc((void **)&dev_a, (size_t)m * n * sizeof(float));
        if (err == cudaSuccess) err = cudaMalloc((void **)&dev_b, (size_t)n * p * sizeof(float));
        if (err == cudaSuccess) err = cudaMalloc((void **)&dev_c, (size_t)m * p * sizeof(float));
        if (err == cudaSuccess) err = cudaMemcpy(dev_a, host_a, (size_t)m * n * sizeof(float), cudaMemcpyHostToDevice);
        if (err == cudaSuccess) err = cudaMemcpy(dev_b, host_b, (size_t)n * p * sizeof(float), cudaMemcpyHostToDevice);
        if (err == cudaSuccess) err = cuda_matrix_multiply_kernel_wrapper(dev_a, dev_b, dev_c, (int)m, (int)n, (int)p, 0);
        if (err == cudaSuccess) err = cudaMemcpy(host_c, dev_c, (size_t)m * p * sizeof(float), cudaMemcpyDeviceToHost);

        if (dev_a) cudaFree(dev_a);
        if (dev_b) cudaFree(dev_b);
        if (dev_c) cudaFree(dev_c);
    } else if (CUDA_G(enable_cpu_fallback)) {
        cpu_matrix_multiply(host_a, host_b, host_c, (int)m, (int)n, (int)p);
    } else {
        err = cudaErrorNoDevice;
    }

    efree(host_a);
    efree(host_b);

    if (err != cudaSuccess) {
        efree(host_c);
        cuda_report_error(err, __FILE__, __LINE__);
        RETURN_FALSE;
    }

    /* Build the nested result array. */
    zval_ptr_dtor(result);
    array_init_size(result, (uint32_t)m);
    for (zend_long i = 0; i < m; i++) {
        zval out_row;
        array_init_size(&out_row, (uint32_t)p);
        for (zend_long j = 0; j < p; j++) {
            add_next_index_double(&out_row, (double)host_c[i * p + j]);
        }
        add_next_index_zval(result, &out_row);
    }

    efree(host_c);
    RETURN_TRUE;
}

/* -------------------------------------------------------------------------
 * Error handling
 * ------------------------------------------------------------------------- */
PHP_FUNCTION(cuda_get_last_error) {
    ZEND_PARSE_PARAMETERS_NONE();
    RETURN_LONG((zend_long)cudaGetLastError());
}

PHP_FUNCTION(cuda_get_error_string) {
    zend_long code;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_LONG(code)
    ZEND_PARSE_PARAMETERS_END();
    RETURN_STRING(cudaGetErrorString((cudaError_t)code));
}

PHP_FUNCTION(cuda_get_error_name) {
    zend_long code;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_LONG(code)
    ZEND_PARSE_PARAMETERS_END();
    RETURN_STRING(cudaGetErrorName((cudaError_t)code));
}

/* -------------------------------------------------------------------------
 * Profiling control
 * ------------------------------------------------------------------------- */
PHP_FUNCTION(cuda_profiler_start) {
    ZEND_PARSE_PARAMETERS_NONE();
    CUDA_CHECK_RET(cudaProfilerStart());
    RETURN_TRUE;
}

PHP_FUNCTION(cuda_profiler_stop) {
    ZEND_PARSE_PARAMETERS_NONE();
    CUDA_CHECK_RET(cudaProfilerStop());
    RETURN_TRUE;
}

/* -------------------------------------------------------------------------
 * Function table
 * ------------------------------------------------------------------------- */
ZEND_BEGIN_ARG_INFO_EX(arginfo_void, 0, 0, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_device_id, 0, 0, 1)
    ZEND_ARG_TYPE_INFO(0, device_id, IS_LONG, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_size, 0, 0, 1)
    ZEND_ARG_TYPE_INFO(0, size, IS_LONG, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_resource, 0, 0, 1)
    ZEND_ARG_INFO(0, resource)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_memset, 0, 0, 2)
    ZEND_ARG_INFO(0, resource)
    ZEND_ARG_TYPE_INFO(0, value, IS_LONG, 0)
    ZEND_ARG_TYPE_INFO(0, size, IS_LONG, 1)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_htod, 0, 0, 2)
    ZEND_ARG_INFO(0, resource)
    ZEND_ARG_TYPE_INFO(0, data, IS_STRING, 0)
    ZEND_ARG_TYPE_INFO(0, offset, IS_LONG, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_dtoh, 0, 0, 1)
    ZEND_ARG_INFO(0, resource)
    ZEND_ARG_TYPE_INFO(0, size, IS_LONG, 0)
    ZEND_ARG_TYPE_INFO(0, offset, IS_LONG, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_dtod, 0, 0, 2)
    ZEND_ARG_INFO(0, dst)
    ZEND_ARG_INFO(0, src)
    ZEND_ARG_TYPE_INFO(0, size, IS_LONG, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_matrix_multiply, 0, 0, 3)
    ZEND_ARG_TYPE_INFO(0, matrix_a, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO(0, matrix_b, IS_ARRAY, 0)
    ZEND_ARG_INFO(1, result)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_error_code, 0, 0, 1)
    ZEND_ARG_TYPE_INFO(0, error_code, IS_LONG, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_pool_allocate, 0, 0, 2)
    ZEND_ARG_INFO(0, pool)
    ZEND_ARG_TYPE_INFO(0, size, IS_LONG, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_pool_free, 0, 0, 2)
    ZEND_ARG_INFO(0, pool)
    ZEND_ARG_INFO(0, memory)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_stream_wait_event, 0, 0, 2)
    ZEND_ARG_INFO(0, stream)
    ZEND_ARG_INFO(0, event)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_stream_opt, 0, 0, 0)
    ZEND_ARG_INFO(0, stream)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_event_record, 0, 0, 1)
    ZEND_ARG_INFO(0, event)
    ZEND_ARG_INFO(0, stream)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_graph_launch, 0, 0, 1)
    ZEND_ARG_INFO(0, graph)
    ZEND_ARG_INFO(0, stream)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_cublas_mm, 0, 0, 7)
    ZEND_ARG_INFO(0, handle)
    ZEND_ARG_TYPE_INFO(0, matrix_a, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO(0, matrix_b, IS_ARRAY, 0)
    ZEND_ARG_INFO(1, result)
    ZEND_ARG_TYPE_INFO(0, m, IS_LONG, 0)
    ZEND_ARG_TYPE_INFO(0, n, IS_LONG, 0)
    ZEND_ARG_TYPE_INFO(0, k, IS_LONG, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_cublas_gemm, 0, 0, 9)
    ZEND_ARG_INFO(0, handle)
    ZEND_ARG_TYPE_INFO(0, matrix_a, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO(0, matrix_b, IS_ARRAY, 0)
    ZEND_ARG_INFO(1, result)
    ZEND_ARG_TYPE_INFO(0, m, IS_LONG, 0)
    ZEND_ARG_TYPE_INFO(0, n, IS_LONG, 0)
    ZEND_ARG_TYPE_INFO(0, k, IS_LONG, 0)
    ZEND_ARG_TYPE_INFO(0, alpha, IS_DOUBLE, 0)
    ZEND_ARG_TYPE_INFO(0, beta, IS_DOUBLE, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_batch_gemm, 0, 0, 8)
    ZEND_ARG_INFO(0, handle)
    ZEND_ARG_TYPE_INFO(0, matrices_a, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO(0, matrices_b, IS_ARRAY, 0)
    ZEND_ARG_INFO(1, results)
    ZEND_ARG_TYPE_INFO(0, m, IS_LONG, 0)
    ZEND_ARG_TYPE_INFO(0, n, IS_LONG, 0)
    ZEND_ARG_TYPE_INFO(0, k, IS_LONG, 0)
    ZEND_ARG_TYPE_INFO(0, batch_size, IS_LONG, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_cudnn_conv_fwd, 0, 0, 11)
    ZEND_ARG_TYPE_INFO(0, input, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO(0, filter, IS_ARRAY, 0)
    ZEND_ARG_INFO(1, output)
    ZEND_ARG_TYPE_INFO(0, batch_size, IS_LONG, 0)
    ZEND_ARG_TYPE_INFO(0, in_channels, IS_LONG, 0)
    ZEND_ARG_TYPE_INFO(0, height, IS_LONG, 0)
    ZEND_ARG_TYPE_INFO(0, width, IS_LONG, 0)
    ZEND_ARG_TYPE_INFO(0, filter_count, IS_LONG, 0)
    ZEND_ARG_TYPE_INFO(0, filter_height, IS_LONG, 0)
    ZEND_ARG_TYPE_INFO(0, filter_width, IS_LONG, 0)
    ZEND_ARG_TYPE_INFO(0, stride, IS_LONG, 0)
    ZEND_ARG_TYPE_INFO(0, padding, IS_LONG, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_kernel_compile, 0, 0, 2)
    ZEND_ARG_TYPE_INFO(0, source, IS_STRING, 0)
    ZEND_ARG_TYPE_INFO(0, kernel_name, IS_STRING, 0)
    ZEND_ARG_TYPE_INFO(0, options, IS_ARRAY, 1)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_kernel_launch, 0, 0, 4)
    ZEND_ARG_INFO(0, kernel)
    ZEND_ARG_TYPE_INFO(0, args, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO(0, grid, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO(0, block, IS_ARRAY, 0)
    ZEND_ARG_INFO(0, stream)
ZEND_END_ARG_INFO()

static const zend_function_entry cuda_functions[] = {
    /* Device */
    PHP_FE(cuda_device_count, arginfo_void)
    PHP_FE(cuda_device_properties, arginfo_device_id)
    PHP_FE(cuda_set_device, arginfo_device_id)
    PHP_FE(cuda_get_device, arginfo_void)
    PHP_FE(cuda_device_reset, arginfo_void)
    PHP_FE(cuda_device_synchronize, arginfo_void)
    PHP_FE(cuda_driver_version, arginfo_void)
    PHP_FE(cuda_runtime_version, arginfo_void)

    /* Memory */
    PHP_FE(cuda_malloc, arginfo_size)
    PHP_FE(cuda_free, arginfo_resource)
    PHP_FE(cuda_memset, arginfo_memset)
    PHP_FE(cuda_memcpy_host_to_device, arginfo_htod)
    PHP_FE(cuda_memcpy_device_to_host, arginfo_dtoh)
    PHP_FE(cuda_memcpy_device_to_device, arginfo_dtod)
    PHP_FE(cuda_pinned_alloc, arginfo_size)
    PHP_FE(cuda_unified_alloc, arginfo_size)
    PHP_FE(cuda_memory_get_info, arginfo_void)
    PHP_FE(cuda_measure_memory_bandwidth, arginfo_size)

    /* Memory pool */
    PHP_FE(cuda_memory_pool_init, arginfo_size)
    PHP_FE(cuda_memory_pool_destroy, arginfo_resource)
    PHP_FE(cuda_memory_pool_allocate, arginfo_pool_allocate)
    PHP_FE(cuda_memory_pool_free, arginfo_pool_free)
    PHP_FE(cuda_memory_pool_stats, arginfo_resource)

    /* Compute */
    PHP_FE(cuda_matrix_multiply, arginfo_matrix_multiply)

    /* Errors */
    PHP_FE(cuda_get_last_error, arginfo_void)
    PHP_FE(cuda_get_error_string, arginfo_error_code)
    PHP_FE(cuda_get_error_name, arginfo_error_code)

    /* Profiling */
    PHP_FE(cuda_profiler_start, arginfo_void)
    PHP_FE(cuda_profiler_stop, arginfo_void)

    /* Streams, events, graphs (streams.c) */
    PHP_FE(cuda_stream_create, arginfo_void)
    PHP_FE(cuda_stream_destroy, arginfo_resource)
    PHP_FE(cuda_stream_synchronize, arginfo_resource)
    PHP_FE(cuda_stream_query, arginfo_resource)
    PHP_FE(cuda_stream_wait_event, arginfo_stream_wait_event)
    PHP_FE(cuda_event_create, arginfo_void)
    PHP_FE(cuda_event_destroy, arginfo_resource)
    PHP_FE(cuda_event_record_start, arginfo_event_record)
    PHP_FE(cuda_event_record_stop, arginfo_event_record)
    PHP_FE(cuda_event_elapsed_time, arginfo_resource)
    PHP_FE(cuda_graph_begin_capture, arginfo_stream_opt)
    PHP_FE(cuda_graph_end_capture, arginfo_void)
    PHP_FE(cuda_graph_launch, arginfo_graph_launch)
    PHP_FE(cuda_graph_destroy, arginfo_resource)

    /* cuBLAS (cublas_ops.c) */
    PHP_FE(cuda_cublas_create, arginfo_void)
    PHP_FE(cuda_cublas_destroy, arginfo_resource)
    PHP_FE(cuda_cublas_matrix_multiply, arginfo_cublas_mm)
    PHP_FE(cuda_cublas_gemm, arginfo_cublas_gemm)
    PHP_FE(cuda_batch_gemm, arginfo_batch_gemm)

#ifdef HAVE_CUDNN
    PHP_FE(cuda_cudnn_convolution_forward, arginfo_cudnn_conv_fwd)
#endif
#ifdef HAVE_NVRTC
    PHP_FE(cuda_kernel_compile, arginfo_kernel_compile)
    PHP_FE(cuda_kernel_launch, arginfo_kernel_launch)
#endif
    PHP_FE_END
};

zend_module_entry cuda_module_entry = {
    STANDARD_MODULE_HEADER,
    PHP_CUDA_EXTNAME,
    cuda_functions,
    PHP_MINIT(cuda),
    PHP_MSHUTDOWN(cuda),
    PHP_RINIT(cuda),
    PHP_RSHUTDOWN(cuda),
    PHP_MINFO(cuda),
    PHP_CUDA_VERSION,
    STANDARD_MODULE_PROPERTIES
};

#ifdef COMPILE_DL_CUDA
ZEND_GET_MODULE(cuda)
#endif
