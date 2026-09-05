#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#ifdef HAVE_NVRTC

#include "php.h"
#include "zend_exceptions.h"
#include "php_cuda.h"
#include "tensor.h"

#include <nvrtc.h>
#include <cuda.h>

typedef struct _cuda_kernel_resource {
    CUmodule module;
    CUfunction function;
    char *name;
    int device_id;
} cuda_kernel_resource;

void php_cuda_kernel_dtor(zend_resource *rsrc) {
    cuda_kernel_resource *res = (cuda_kernel_resource *)rsrc->ptr;
    if (!res) return;
    cudaSetDevice(res->device_id);
    cuModuleUnload(res->module);
    efree(res->name);
    efree(res);
}

static void nvrtc_report(nvrtcResult r, nvrtcProgram prog, const char *what) {
    char *msg = NULL;
    if (prog) {
        size_t log_size = 0;
        if (nvrtcGetProgramLogSize(prog, &log_size) == NVRTC_SUCCESS && log_size > 1) {
            msg = emalloc(log_size);
            nvrtcGetProgramLog(prog, msg);
        }
    }
    if (msg) {
        zend_throw_exception_ex(cuda_exception_ce, (zend_long)r, "%s: %s\n%s", what, nvrtcGetErrorString(r), msg);
        efree(msg);
    } else {
        zend_throw_exception_ex(cuda_exception_ce, (zend_long)r, "%s: %s", what, nvrtcGetErrorString(r));
    }
}

static zend_bool php_cuda_driver_initialized = 0;

static int php_cuda_ensure_driver(void) {
    if (!php_cuda_driver_initialized) {
        if (cuInit(0) != CUDA_SUCCESS) return FAILURE;
        php_cuda_driver_initialized = 1;
    }
    return SUCCESS;
}

PHP_FUNCTION(cuda_kernel_compile) {
    zend_string *source, *kernel_name;
    zval *options_zv = NULL;
    ZEND_PARSE_PARAMETERS_START(2, 3)
        Z_PARAM_STR(source)
        Z_PARAM_STR(kernel_name)
        Z_PARAM_OPTIONAL
        Z_PARAM_ARRAY_OR_NULL(options_zv)
    ZEND_PARSE_PARAMETERS_END();

    if (php_cuda_ensure_driver() == FAILURE) {
        cuda_report_error_msg("cuda_kernel_compile: failed to initialize the CUDA driver API");
        RETURN_FALSE;
    }
    CUDA_CHECK_RET(cuda_use_device(CUDA_G(current_device)));

    nvrtcProgram prog;
    nvrtcResult r = nvrtcCreateProgram(&prog, ZSTR_VAL(source), "php_cuda_kernel.cu", 0, NULL, NULL);
    if (r != NVRTC_SUCCESS) {
        nvrtc_report(r, NULL, "nvrtcCreateProgram failed");
        RETURN_FALSE;
    }

    /* Default options: C++17 and the current device's architecture. */
    char arch_opt[64] = {0};
    int device = CUDA_G(current_device);
    cudaDeviceProp props;
    if (cudaGetDeviceProperties(&props, device) == cudaSuccess) {
        snprintf(arch_opt, sizeof(arch_opt), "--gpu-architecture=compute_%d%d", props.major, props.minor);
    }

    const char *opts[32];
    int nopts = 0;
    opts[nopts++] = "--std=c++17";
    if (arch_opt[0]) opts[nopts++] = arch_opt;

    if (options_zv) {
        zval *zv;
        ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(options_zv), zv) {
            if (nopts >= 30) break;
            if (Z_TYPE_P(zv) == IS_STRING) {
                opts[nopts++] = Z_STRVAL_P(zv);
            }
        } ZEND_HASH_FOREACH_END();
    }

    r = nvrtcCompileProgram(prog, nopts, opts);
    if (r != NVRTC_SUCCESS) {
        nvrtc_report(r, prog, "Kernel compilation failed");
        nvrtcDestroyProgram(&prog);
        RETURN_FALSE;
    }

    size_t ptx_size = 0;
    r = nvrtcGetPTXSize(prog, &ptx_size);
    if (r != NVRTC_SUCCESS || ptx_size == 0) {
        nvrtc_report(r, prog, "nvrtcGetPTXSize failed");
        nvrtcDestroyProgram(&prog);
        RETURN_FALSE;
    }

    char *ptx = emalloc(ptx_size);
    r = nvrtcGetPTX(prog, ptx);
    nvrtcDestroyProgram(&prog);
    if (r != NVRTC_SUCCESS) {
        efree(ptx);
        nvrtc_report(r, NULL, "nvrtcGetPTX failed");
        RETURN_FALSE;
    }

    cuda_kernel_resource *res = ecalloc(1, sizeof(cuda_kernel_resource));
    res->device_id = device;

    CUresult cr = cuModuleLoadData(&res->module, ptx);
    efree(ptx);
    if (cr != CUDA_SUCCESS) {
        const char *err_str = "unknown";
        cuGetErrorString(cr, &err_str);
        zend_throw_exception_ex(cuda_exception_ce, (zend_long)cr, "cuModuleLoadData failed: %s", err_str);
        efree(res);
        RETURN_FALSE;
    }

    cr = cuModuleGetFunction(&res->function, res->module, ZSTR_VAL(kernel_name));
    if (cr != CUDA_SUCCESS) {
        const char *err_str = "unknown";
        cuGetErrorString(cr, &err_str);
        zend_throw_exception_ex(cuda_exception_ce, (zend_long)cr,
            "Kernel '%s' not found in module: %s", ZSTR_VAL(kernel_name), err_str);
        cuModuleUnload(res->module);
        efree(res);
        RETURN_FALSE;
    }

    res->name = estrndup(ZSTR_VAL(kernel_name), ZSTR_LEN(kernel_name));
    RETURN_RES(zend_register_resource(res, le_cuda_kernel));
}

PHP_FUNCTION(cuda_kernel_launch) {
    zval *kernel_zv, *args_zv, *grid_zv, *block_zv, *stream_zv = NULL;
    ZEND_PARSE_PARAMETERS_START(4, 5)
        Z_PARAM_RESOURCE(kernel_zv)
        Z_PARAM_ARRAY(args_zv)
        Z_PARAM_ARRAY(grid_zv)
        Z_PARAM_ARRAY(block_zv)
        Z_PARAM_OPTIONAL
        Z_PARAM_RESOURCE_OR_NULL(stream_zv)
    ZEND_PARSE_PARAMETERS_END();

    cuda_kernel_resource *kernel = (cuda_kernel_resource *)zend_fetch_resource(
        Z_RES_P(kernel_zv), "CUDA Kernel", le_cuda_kernel);
    if (!kernel) RETURN_FALSE;

    /* Grid / block dimensions. */
    zend_long grid[3] = {1, 1, 1};
    zend_long block[3] = {1, 1, 1};
    int i = 0;
    zval *zv;
    ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(grid_zv), zv) {
        if (i >= 3) break;
        grid[i++] = zval_get_long(zv);
    } ZEND_HASH_FOREACH_END();
    i = 0;
    ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(block_zv), zv) {
        if (i >= 3) break;
        block[i++] = zval_get_long(zv);
    } ZEND_HASH_FOREACH_END();

    if (grid[0] <= 0 || grid[1] <= 0 || grid[2] <= 0 ||
        block[0] <= 0 || block[1] <= 0 || block[2] <= 0) {
        cuda_report_error_msg("cuda_kernel_launch: grid and block dimensions must be positive");
        RETURN_FALSE;
    }

    /* Marshal arguments. Values must outlive the launch call. */
    HashTable *args = Z_ARRVAL_P(args_zv);
    uint32_t nargs = zend_hash_num_elements(args);

    void **kernel_params = ecalloc(nargs ? nargs : 1, sizeof(void *));
    CUdeviceptr *ptr_values = ecalloc(nargs ? nargs : 1, sizeof(CUdeviceptr));
    int64_t *int_values = ecalloc(nargs ? nargs : 1, sizeof(int64_t));
    double *float_values = ecalloc(nargs ? nargs : 1, sizeof(double));

    uint32_t argi = 0;
    zend_bool args_ok = 1;
    ZEND_HASH_FOREACH_VAL(args, zv) {
        if (Z_TYPE_P(zv) == IS_OBJECT && Z_OBJCE_P(zv) == cuda_tensor_ce) {
            ct_obj *t = ct_obj_from_zval(zv);
            ptr_values[argi] = (CUdeviceptr)((char *)t->storage->data +
                (size_t)t->offset * ct_dtype_size(t->dtype));
            kernel_params[argi] = &ptr_values[argi];
        } else if (Z_TYPE_P(zv) == IS_LONG) {
            int_values[argi] = (int64_t)Z_LVAL_P(zv);
            kernel_params[argi] = &int_values[argi];
        } else if (Z_TYPE_P(zv) == IS_DOUBLE) {
            float_values[argi] = Z_DVAL_P(zv);
            kernel_params[argi] = &float_values[argi];
        } else {
            args_ok = 0;
            break;
        }
        argi++;
    } ZEND_HASH_FOREACH_END();

    if (!args_ok) {
        efree(kernel_params);
        efree(ptr_values);
        efree(int_values);
        efree(float_values);
        cuda_report_error_msg("cuda_kernel_launch: arguments must be CudaTensor objects, ints, or floats");
        RETURN_FALSE;
    }

    /* Optional stream. */
    CUstream stream = (CUstream)0;
    if (stream_zv) {
        cuda_stream_resource *sres = (cuda_stream_resource *)zend_fetch_resource(
            Z_RES_P(stream_zv), "CUDA Stream", le_cuda_stream);
        if (!sres) {
            efree(kernel_params);
            efree(ptr_values);
            efree(int_values);
            efree(float_values);
            RETURN_FALSE;
        }
        stream = (CUstream)sres->stream;
    }

    CUDA_CHECK_RET(cuda_use_device(kernel->device_id));

    CUresult cr = cuLaunchKernel(kernel->function,
                                 (unsigned)grid[0], (unsigned)grid[1], (unsigned)grid[2],
                                 (unsigned)block[0], (unsigned)block[1], (unsigned)block[2],
                                 0, stream, kernel_params, NULL);

    efree(kernel_params);
    efree(ptr_values);
    efree(int_values);
    efree(float_values);

    if (cr != CUDA_SUCCESS) {
        const char *err_str = "unknown";
        cuGetErrorString(cr, &err_str);
        zend_throw_exception_ex(cuda_exception_ce, (zend_long)cr, "cuLaunchKernel failed: %s", err_str);
        RETURN_FALSE;
    }

    RETURN_TRUE;
}

#endif /* HAVE_NVRTC */
