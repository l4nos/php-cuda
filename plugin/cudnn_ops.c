#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#ifdef HAVE_CUDNN

#include "php.h"
#include "zend_exceptions.h"
#include "php_cuda.h"
#include <cudnn.h>

/* Per-device cached cuDNN handles. */
#define PHP_CUDNN_MAX_DEVICES 64
static cudnnHandle_t php_cudnn_handles[PHP_CUDNN_MAX_DEVICES] = {NULL};

static cudnnHandle_t php_cuda_get_cudnn_handle(int device_id) {
    if (device_id < 0 || device_id >= PHP_CUDNN_MAX_DEVICES) return NULL;
    if (!php_cudnn_handles[device_id]) {
        if (cuda_use_device(device_id) != cudaSuccess) return NULL;
        if (cudnnCreate(&php_cudnn_handles[device_id]) != CUDNN_STATUS_SUCCESS) {
            return NULL;
        }
    }
    return php_cudnn_handles[device_id];
}

static int cudnn_fail(cudnnStatus_t status, const char *what) {
    char buf[256];
    snprintf(buf, sizeof(buf), "%s: %s", what, cudnnGetErrorString(status));
    cuda_report_error_msg(buf);
    return FAILURE;
}

/*
 * cuda_cudnn_convolution_forward(
 *     array input, array filter, &output,
 *     int batch, int in_channels, int height, int width,
 *     int filter_count, int filter_height, int filter_width,
 *     int stride, int padding)
 *
 * NCHW fp32 input, OIHW fp32 filter. Returns the output as a flat array and
 * its shape via the output argument: ['shape' => [n,c,h,w], 'data' => [...]].
 */
PHP_FUNCTION(cuda_cudnn_convolution_forward) {
    zval *input_zv, *filter_zv, *output_zv;
    zend_long batch, in_channels, height, width;
    zend_long filter_count, filter_height, filter_width, stride, padding;

    ZEND_PARSE_PARAMETERS_START(12, 12)
        Z_PARAM_ARRAY(input_zv)
        Z_PARAM_ARRAY(filter_zv)
        Z_PARAM_ZVAL(output_zv)
        Z_PARAM_LONG(batch)
        Z_PARAM_LONG(in_channels)
        Z_PARAM_LONG(height)
        Z_PARAM_LONG(width)
        Z_PARAM_LONG(filter_count)
        Z_PARAM_LONG(filter_height)
        Z_PARAM_LONG(filter_width)
        Z_PARAM_LONG(stride)
        Z_PARAM_LONG(padding)
    ZEND_PARSE_PARAMETERS_END();

    if (batch <= 0 || in_channels <= 0 || height <= 0 || width <= 0 ||
        filter_count <= 0 || filter_height <= 0 || filter_width <= 0 ||
        stride <= 0 || padding < 0) {
        cuda_report_error_msg("cuda_cudnn_convolution_forward: invalid dimensions");
        RETURN_FALSE;
    }

    int device_id = CUDA_G(current_device);
    cudnnHandle_t handle = php_cuda_get_cudnn_handle(device_id);
    if (!handle) {
        cuda_report_error_msg("cuda_cudnn_convolution_forward: failed to create cuDNN handle");
        RETURN_FALSE;
    }

    /* Validate and flatten inputs. */
    zend_long input_count, filter_elems;
    float *host_input = php_cuda_array_to_floats(input_zv, &input_count);
    float *host_filter = php_cuda_array_to_floats(filter_zv, &filter_elems);

    zend_long expected_input = batch * in_channels * height * width;
    zend_long expected_filter = filter_count * in_channels * filter_height * filter_width;
    if (input_count != expected_input || filter_elems != expected_filter) {
        efree(host_input);
        efree(host_filter);
        cuda_report_error_msg("cuda_cudnn_convolution_forward: input/filter sizes do not match the given dimensions");
        RETURN_FALSE;
    }

    cudnnTensorDescriptor_t input_desc = NULL, output_desc = NULL;
    cudnnFilterDescriptor_t filter_desc = NULL;
    cudnnConvolutionDescriptor_t conv_desc = NULL;
    void *dev_input = NULL, *dev_filter = NULL, *dev_output = NULL, *workspace = NULL;
    float *host_output = NULL;
    cudnnStatus_t st;
    int ok = 0;

    st = cudnnCreateTensorDescriptor(&input_desc);
    if (st == CUDNN_STATUS_SUCCESS) st = cudnnCreateTensorDescriptor(&output_desc);
    if (st == CUDNN_STATUS_SUCCESS) st = cudnnCreateFilterDescriptor(&filter_desc);
    if (st == CUDNN_STATUS_SUCCESS) st = cudnnCreateConvolutionDescriptor(&conv_desc);
    if (st != CUDNN_STATUS_SUCCESS) {
        cudnn_fail(st, "descriptor creation failed");
        goto cleanup;
    }

    st = cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                    (int)batch, (int)in_channels, (int)height, (int)width);
    if (st == CUDNN_STATUS_SUCCESS) {
        st = cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                        (int)filter_count, (int)in_channels,
                                        (int)filter_height, (int)filter_width);
    }
    if (st == CUDNN_STATUS_SUCCESS) {
        st = cudnnSetConvolution2dDescriptor(conv_desc,
                                             (int)padding, (int)padding,
                                             (int)stride, (int)stride,
                                             1, 1,
                                             CUDNN_CROSS_CORRELATION,
                                             CUDNN_DATA_FLOAT);
    }
    if (st != CUDNN_STATUS_SUCCESS) {
        cudnn_fail(st, "descriptor setup failed");
        goto cleanup;
    }

    int out_n = 0, out_c = 0, out_h = 0, out_w = 0;
    st = cudnnGetConvolution2dForwardOutputDim(conv_desc, input_desc, filter_desc,
                                               &out_n, &out_c, &out_h, &out_w);
    if (st != CUDNN_STATUS_SUCCESS) {
        cudnn_fail(st, "failed to compute output dimensions");
        goto cleanup;
    }

    st = cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                    out_n, out_c, out_h, out_w);
    if (st != CUDNN_STATUS_SUCCESS) {
        cudnn_fail(st, "output descriptor setup failed");
        goto cleanup;
    }

    /* Algorithm selection: the legacy cudnnGetConvolutionForwardAlgorithm was
     * removed in cuDNN 9; the _v7 API exists on 8 and 9. */
    cudnnConvolutionFwdAlgo_t algo;
#if CUDNN_MAJOR >= 8
    int algo_count = 0;
    cudnnConvolutionFwdAlgoPerf_t perf;
    st = cudnnGetConvolutionForwardAlgorithm_v7(handle, input_desc, filter_desc,
                                                conv_desc, output_desc, 1,
                                                &algo_count, &perf);
    algo = (st == CUDNN_STATUS_SUCCESS && algo_count > 0) ? perf.algo
                                                          : CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    if (st != CUDNN_STATUS_SUCCESS) {
        cudnn_fail(st, "algorithm selection failed");
        goto cleanup;
    }
#else
    st = cudnnGetConvolutionForwardAlgorithm(handle, input_desc, filter_desc,
                                             conv_desc, output_desc,
                                             CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo);
    if (st != CUDNN_STATUS_SUCCESS) {
        cudnn_fail(st, "algorithm selection failed");
        goto cleanup;
    }
#endif

    size_t workspace_size = 0;
    st = cudnnGetConvolutionForwardWorkspaceSize(handle, input_desc, filter_desc,
                                                 conv_desc, output_desc, algo,
                                                 &workspace_size);
    if (st != CUDNN_STATUS_SUCCESS) {
        cudnn_fail(st, "workspace size query failed");
        goto cleanup;
    }

    size_t input_bytes = (size_t)expected_input * sizeof(float);
    size_t filter_bytes = (size_t)expected_filter * sizeof(float);
    size_t output_elems = (size_t)out_n * out_c * out_h * out_w;
    size_t output_bytes = output_elems * sizeof(float);

    if (cudaMalloc(&dev_input, input_bytes) != cudaSuccess ||
        cudaMalloc(&dev_filter, filter_bytes) != cudaSuccess ||
        cudaMalloc(&dev_output, output_bytes) != cudaSuccess ||
        (workspace_size > 0 && cudaMalloc(&workspace, workspace_size) != cudaSuccess)) {
        cuda_report_error_msg("cuda_cudnn_convolution_forward: device allocation failed");
        goto cleanup;
    }

    if (cudaMemcpy(dev_input, host_input, input_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(dev_filter, host_filter, filter_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cuda_report_error_msg("cuda_cudnn_convolution_forward: host-to-device copy failed");
        goto cleanup;
    }

    {
        float alpha = 1.0f, beta = 0.0f;
        st = cudnnConvolutionForward(handle, &alpha,
                                     input_desc, dev_input,
                                     filter_desc, dev_filter,
                                     conv_desc, algo,
                                     workspace, workspace_size,
                                     &beta,
                                     output_desc, dev_output);
    }
    if (st != CUDNN_STATUS_SUCCESS) {
        cudnn_fail(st, "cudnnConvolutionForward failed");
        goto cleanup;
    }

    host_output = emalloc(output_bytes);
    if (cudaMemcpy(host_output, dev_output, output_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        cuda_report_error_msg("cuda_cudnn_convolution_forward: device-to-host copy failed");
        goto cleanup;
    }

    /* ['shape' => [n,c,h,w], 'data' => flat array] */
    {
        zval shape, data;
        array_init_size(&shape, 4);
        add_next_index_long(&shape, out_n);
        add_next_index_long(&shape, out_c);
        add_next_index_long(&shape, out_h);
        add_next_index_long(&shape, out_w);
        php_cuda_floats_to_array(host_output, (zend_long)output_elems, &data);

        zval_ptr_dtor(output_zv);
        array_init(output_zv);
        add_assoc_zval(output_zv, "shape", &shape);
        add_assoc_zval(output_zv, "data", &data);
    }
    ok = 1;

cleanup:
    if (host_output) efree(host_output);
    if (dev_input) cudaFree(dev_input);
    if (dev_filter) cudaFree(dev_filter);
    if (dev_output) cudaFree(dev_output);
    if (workspace) cudaFree(workspace);
    if (input_desc) cudnnDestroyTensorDescriptor(input_desc);
    if (output_desc) cudnnDestroyTensorDescriptor(output_desc);
    if (filter_desc) cudnnDestroyFilterDescriptor(filter_desc);
    if (conv_desc) cudnnDestroyConvolutionDescriptor(conv_desc);
    efree(host_input);
    efree(host_filter);

    if (!ok) {
        RETURN_FALSE;
    }
    RETURN_TRUE;
}

#endif /* HAVE_CUDNN */
