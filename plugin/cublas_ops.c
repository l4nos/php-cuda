#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "php.h"
#include "php_cuda.h"

/* -------------------------------------------------------------------------
 * Resource destructor (registered in cuda.c MINIT)
 * ------------------------------------------------------------------------- */
void php_cuda_cublas_dtor(zend_resource *rsrc) {
    cuda_cublas_resource *res = (cuda_cublas_resource *)rsrc->ptr;
    if (!res) return;
    cudaSetDevice(res->device_id);
    cublasDestroy(res->handle);
    efree(res);
}

static cuda_cublas_resource *fetch_cublas(zval *zv) {
    return (cuda_cublas_resource *)zend_fetch_resource(Z_RES_P(zv), "cuBLAS Handle", le_cublas_handle);
}

/* -------------------------------------------------------------------------
 * Handle management
 * ------------------------------------------------------------------------- */
PHP_FUNCTION(cuda_cublas_create) {
    ZEND_PARSE_PARAMETERS_NONE();

    CUDA_CHECK_RET(cuda_use_device(CUDA_G(current_device)));

    cuda_cublas_resource *res = emalloc(sizeof(cuda_cublas_resource));
    res->device_id = CUDA_G(current_device);
    if (cublasCreate(&res->handle) != CUBLAS_STATUS_SUCCESS) {
        efree(res);
        cuda_report_error_msg("cuda_cublas_create: failed to create cuBLAS handle");
        RETURN_FALSE;
    }

    RETURN_RES(zend_register_resource(res, le_cublas_handle));
}

PHP_FUNCTION(cuda_cublas_destroy) {
    zval *res;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_RESOURCE(res)
    ZEND_PARSE_PARAMETERS_END();

    if (!fetch_cublas(res)) RETURN_FALSE;
    zend_list_close(Z_RES_P(res));
    RETURN_TRUE;
}

/* -------------------------------------------------------------------------
 * Shared GEMM implementation.
 *
 * PHP arrays are row-major; cuBLAS is column-major. For row-major
 * C(m x n) = A(m x k) * B(k x n) we compute C^T = B^T * A^T by swapping the
 * operands: sgemm(n, m, k, B, n, A, k, C, n). No transposition cost.
 * ------------------------------------------------------------------------- */
static int php_cublas_gemm_impl(cuda_cublas_resource *res,
                                zval *a_zv, zval *b_zv, zval *result_zv,
                                zend_long m, zend_long n, zend_long k,
                                float alpha, float beta) {
    if (m <= 0 || n <= 0 || k <= 0) {
        cuda_report_error_msg("cuda_cublas: dimensions must be positive");
        return FAILURE;
    }

    zend_long a_count, b_count;
    float *host_a = php_cuda_array_to_floats(a_zv, &a_count);
    float *host_b = php_cuda_array_to_floats(b_zv, &b_count);

    if (a_count != m * k || b_count != k * n) {
        efree(host_a);
        efree(host_b);
        cuda_report_error_msg("cuda_cublas: matrix sizes do not match m*n*k");
        return FAILURE;
    }

    /* For beta != 0 the initial C values come from the result array. */
    float *host_c = emalloc((size_t)m * n * sizeof(float));
    if (beta != 0.0f && Z_TYPE_P(result_zv) == IS_ARRAY) {
        zend_long c_count;
        float *initial = php_cuda_array_to_floats(result_zv, &c_count);
        if (c_count == m * n) {
            memcpy(host_c, initial, (size_t)m * n * sizeof(float));
        } else {
            memset(host_c, 0, (size_t)m * n * sizeof(float));
        }
        efree(initial);
    } else {
        memset(host_c, 0, (size_t)m * n * sizeof(float));
    }

    if (cuda_use_device(res->device_id) != cudaSuccess) {
        efree(host_a); efree(host_b); efree(host_c);
        cuda_report_error_msg("cuda_cublas: failed to select device");
        return FAILURE;
    }

    float *dev_a = NULL, *dev_b = NULL, *dev_c = NULL;
    cudaError_t err = cudaSuccess;
    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

    err = cudaMalloc((void **)&dev_a, (size_t)m * k * sizeof(float));
    if (err == cudaSuccess) err = cudaMalloc((void **)&dev_b, (size_t)k * n * sizeof(float));
    if (err == cudaSuccess) err = cudaMalloc((void **)&dev_c, (size_t)m * n * sizeof(float));
    if (err == cudaSuccess) err = cudaMemcpy(dev_a, host_a, (size_t)m * k * sizeof(float), cudaMemcpyHostToDevice);
    if (err == cudaSuccess) err = cudaMemcpy(dev_b, host_b, (size_t)k * n * sizeof(float), cudaMemcpyHostToDevice);
    if (err == cudaSuccess && beta != 0.0f) {
        err = cudaMemcpy(dev_c, host_c, (size_t)m * n * sizeof(float), cudaMemcpyHostToDevice);
    }

    if (err == cudaSuccess) {
        status = cublasSgemm(res->handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             (int)n, (int)m, (int)k,
                             &alpha,
                             dev_b, (int)n,
                             dev_a, (int)k,
                             &beta,
                             dev_c, (int)n);
        if (status == CUBLAS_STATUS_SUCCESS) {
            err = cudaMemcpy(host_c, dev_c, (size_t)m * n * sizeof(float), cudaMemcpyDeviceToHost);
        }
    }

    if (dev_a) cudaFree(dev_a);
    if (dev_b) cudaFree(dev_b);
    if (dev_c) cudaFree(dev_c);

    efree(host_a);
    efree(host_b);

    if (err != cudaSuccess) {
        efree(host_c);
        cuda_report_error(err, __FILE__, __LINE__);
        return FAILURE;
    }
    if (status != CUBLAS_STATUS_SUCCESS) {
        efree(host_c);
        cuda_report_error_msg("cuda_cublas: gemm failed");
        return FAILURE;
    }

    zval_ptr_dtor(result_zv);
    php_cuda_floats_to_array(host_c, m * n, result_zv);
    efree(host_c);
    return SUCCESS;
}

PHP_FUNCTION(cuda_cublas_matrix_multiply) {
    zval *handle_zv, *a_zv, *b_zv, *result_zv;
    zend_long m, n, k;
    ZEND_PARSE_PARAMETERS_START(7, 7)
        Z_PARAM_RESOURCE(handle_zv)
        Z_PARAM_ARRAY(a_zv)
        Z_PARAM_ARRAY(b_zv)
        Z_PARAM_ZVAL(result_zv)
        Z_PARAM_LONG(m)
        Z_PARAM_LONG(n)
        Z_PARAM_LONG(k)
    ZEND_PARSE_PARAMETERS_END();

    cuda_cublas_resource *res = fetch_cublas(handle_zv);
    if (!res) RETURN_FALSE;

    if (php_cublas_gemm_impl(res, a_zv, b_zv, result_zv, m, n, k, 1.0f, 0.0f) == SUCCESS) {
        RETURN_TRUE;
    }
    RETURN_FALSE;
}

PHP_FUNCTION(cuda_cublas_gemm) {
    zval *handle_zv, *a_zv, *b_zv, *result_zv;
    zend_long m, n, k;
    double alpha, beta;
    ZEND_PARSE_PARAMETERS_START(9, 9)
        Z_PARAM_RESOURCE(handle_zv)
        Z_PARAM_ARRAY(a_zv)
        Z_PARAM_ARRAY(b_zv)
        Z_PARAM_ZVAL(result_zv)
        Z_PARAM_LONG(m)
        Z_PARAM_LONG(n)
        Z_PARAM_LONG(k)
        Z_PARAM_DOUBLE(alpha)
        Z_PARAM_DOUBLE(beta)
    ZEND_PARSE_PARAMETERS_END();

    cuda_cublas_resource *res = fetch_cublas(handle_zv);
    if (!res) RETURN_FALSE;

    if (php_cublas_gemm_impl(res, a_zv, b_zv, result_zv, m, n, k,
                             (float)alpha, (float)beta) == SUCCESS) {
        RETURN_TRUE;
    }
    RETURN_FALSE;
}

/* -------------------------------------------------------------------------
 * Batched GEMM: arrays of flat row-major matrices.
 * ------------------------------------------------------------------------- */
PHP_FUNCTION(cuda_batch_gemm) {
    zval *handle_zv, *a_zv, *b_zv, *results_zv;
    zend_long m, n, k, batch_size;
    ZEND_PARSE_PARAMETERS_START(8, 8)
        Z_PARAM_RESOURCE(handle_zv)
        Z_PARAM_ARRAY(a_zv)
        Z_PARAM_ARRAY(b_zv)
        Z_PARAM_ZVAL(results_zv)
        Z_PARAM_LONG(m)
        Z_PARAM_LONG(n)
        Z_PARAM_LONG(k)
        Z_PARAM_LONG(batch_size)
    ZEND_PARSE_PARAMETERS_END();

    cuda_cublas_resource *res = fetch_cublas(handle_zv);
    if (!res) RETURN_FALSE;

    if (m <= 0 || n <= 0 || k <= 0 || batch_size <= 0) {
        cuda_report_error_msg("cuda_batch_gemm: dimensions and batch size must be positive");
        RETURN_FALSE;
    }

    HashTable *ht_a = Z_ARRVAL_P(a_zv);
    HashTable *ht_b = Z_ARRVAL_P(b_zv);
    if (zend_hash_num_elements(ht_a) != (uint32_t)batch_size ||
        zend_hash_num_elements(ht_b) != (uint32_t)batch_size) {
        cuda_report_error_msg("cuda_batch_gemm: input arrays must have batch_size elements");
        RETURN_FALSE;
    }

    if (cuda_use_device(res->device_id) != cudaSuccess) {
        cuda_report_error_msg("cuda_batch_gemm: failed to select device");
        RETURN_FALSE;
    }

    size_t a_elems = (size_t)m * k;
    size_t b_elems = (size_t)k * n;
    size_t c_elems = (size_t)m * n;

    /* Device pointer arrays + one contiguous device buffer per operand. */
    float **h_ptrs_a = emalloc(sizeof(float *) * batch_size);
    float **h_ptrs_b = emalloc(sizeof(float *) * batch_size);
    float **h_ptrs_c = emalloc(sizeof(float *) * batch_size);

    float *dev_a = NULL, *dev_b = NULL, *dev_c = NULL;
    float **dev_ptrs_a = NULL, **dev_ptrs_b = NULL, **dev_ptrs_c = NULL;
    float *host_a = NULL, *host_b = NULL, *host_c = NULL;
    cudaError_t err = cudaSuccess;
    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
    zend_bool ok = 0;

    host_a = emalloc(a_elems * batch_size * sizeof(float));
    host_b = emalloc(b_elems * batch_size * sizeof(float));
    host_c = emalloc(c_elems * batch_size * sizeof(float));

    /* Flatten inputs and validate sizes. */
    zend_long idx = 0;
    zval *mat;
    zend_bool valid = 1;
    ZEND_HASH_FOREACH_VAL(ht_a, mat) {
        if (Z_TYPE_P(mat) != IS_ARRAY) { valid = 0; break; }
        zend_long cnt;
        float *buf = php_cuda_array_to_floats(mat, &cnt);
        if (cnt != (zend_long)a_elems) {
            efree(buf);
            valid = 0;
            break;
        }
        memcpy(host_a + idx * a_elems, buf, a_elems * sizeof(float));
        efree(buf);
        idx++;
    } ZEND_HASH_FOREACH_END();

    if (valid) {
        idx = 0;
        ZEND_HASH_FOREACH_VAL(ht_b, mat) {
            if (Z_TYPE_P(mat) != IS_ARRAY) { valid = 0; break; }
            zend_long cnt;
            float *buf = php_cuda_array_to_floats(mat, &cnt);
            if (cnt != (zend_long)b_elems) {
                efree(buf);
                valid = 0;
                break;
            }
            memcpy(host_b + idx * b_elems, buf, b_elems * sizeof(float));
            efree(buf);
            idx++;
        } ZEND_HASH_FOREACH_END();
    }

    if (!valid) {
        cuda_report_error_msg("cuda_batch_gemm: every matrix must be a flat array of the correct size");
        goto cleanup;
    }

    err = cudaMalloc((void **)&dev_a, a_elems * batch_size * sizeof(float));
    if (err == cudaSuccess) err = cudaMalloc((void **)&dev_b, b_elems * batch_size * sizeof(float));
    if (err == cudaSuccess) err = cudaMalloc((void **)&dev_c, c_elems * batch_size * sizeof(float));
    if (err == cudaSuccess) err = cudaMemcpy(dev_a, host_a, a_elems * batch_size * sizeof(float), cudaMemcpyHostToDevice);
    if (err == cudaSuccess) err = cudaMemcpy(dev_b, host_b, b_elems * batch_size * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup;

    for (zend_long i = 0; i < batch_size; i++) {
        h_ptrs_a[i] = dev_a + i * a_elems;
        h_ptrs_b[i] = dev_b + i * b_elems;
        h_ptrs_c[i] = dev_c + i * c_elems;
    }

    err = cudaMalloc((void **)&dev_ptrs_a, sizeof(float *) * batch_size);
    if (err == cudaSuccess) err = cudaMalloc((void **)&dev_ptrs_b, sizeof(float *) * batch_size);
    if (err == cudaSuccess) err = cudaMalloc((void **)&dev_ptrs_c, sizeof(float *) * batch_size);
    if (err == cudaSuccess) err = cudaMemcpy(dev_ptrs_a, h_ptrs_a, sizeof(float *) * batch_size, cudaMemcpyHostToDevice);
    if (err == cudaSuccess) err = cudaMemcpy(dev_ptrs_b, h_ptrs_b, sizeof(float *) * batch_size, cudaMemcpyHostToDevice);
    if (err == cudaSuccess) err = cudaMemcpy(dev_ptrs_c, h_ptrs_c, sizeof(float *) * batch_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup;

    {
        const float alpha = 1.0f, beta = 0.0f;
        /* Same row-major trick as the single GEMM, per batch element. */
        status = cublasSgemmBatched(res->handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                    (int)n, (int)m, (int)k,
                                    &alpha,
                                    (const float * const *)dev_ptrs_b, (int)n,
                                    (const float * const *)dev_ptrs_a, (int)k,
                                    &beta,
                                    dev_ptrs_c, (int)n,
                                    (int)batch_size);
    }
    if (status != CUBLAS_STATUS_SUCCESS) {
        cuda_report_error_msg("cuda_batch_gemm: cublasSgemmBatched failed");
        goto cleanup;
    }

    err = cudaMemcpy(host_c, dev_c, c_elems * batch_size * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) goto cleanup;

    /* Build results: array of flat arrays. */
    zval_ptr_dtor(results_zv);
    array_init_size(results_zv, (uint32_t)batch_size);
    for (zend_long i = 0; i < batch_size; i++) {
        zval one;
        php_cuda_floats_to_array(host_c + i * c_elems, (zend_long)c_elems, &one);
        add_next_index_zval(results_zv, &one);
    }
    ok = 1;

cleanup:
    if (dev_a) cudaFree(dev_a);
    if (dev_b) cudaFree(dev_b);
    if (dev_c) cudaFree(dev_c);
    if (dev_ptrs_a) cudaFree(dev_ptrs_a);
    if (dev_ptrs_b) cudaFree(dev_ptrs_b);
    if (dev_ptrs_c) cudaFree(dev_ptrs_c);
    if (host_a) efree(host_a);
    if (host_b) efree(host_b);
    if (host_c) efree(host_c);
    efree(h_ptrs_a);
    efree(h_ptrs_b);
    efree(h_ptrs_c);

    if (!ok && err != cudaSuccess) {
        cuda_report_error(err, __FILE__, __LINE__);
    }
    if (!ok) {
        RETURN_FALSE;
    }
    RETURN_TRUE;
}
