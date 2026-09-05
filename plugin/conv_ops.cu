#include "conv_ops.cuh"

extern "C" cudaError_t cuda_batch_convolution_kernel(
    cudnnHandle_t handle,
    const float* const input_arrays[],
    const float* const filter_arrays[],
    float* const output_arrays[],
    int batch_count,
    int batch_size,
    int in_channels,
    int height,
    int width,
    int filter_count,
    int filter_height,
    int filter_width,
    cudaStream_t stream
) {
    cudnnTensorDescriptor_t* input_descs = new cudnnTensorDescriptor_t[batch_count];
    cudnnTensorDescriptor_t* output_descs = new cudnnTensorDescriptor_t[batch_count];
    cudnnFilterDescriptor_t* filter_descs = new cudnnFilterDescriptor_t[batch_count];
    cudnnConvolutionDescriptor_t* conv_descs = new cudnnConvolutionDescriptor_t[batch_count];

    // Honor the caller's stream (default stream when 0)
    cudnnSetStream(handle, stream);
    
    // Initialize descriptors
    for (int i = 0; i < batch_count; i++) {
        cudnnCreateTensorDescriptor(&input_descs[i]);
        cudnnCreateTensorDescriptor(&output_descs[i]);
        cudnnCreateFilterDescriptor(&filter_descs[i]);
        cudnnCreateConvolutionDescriptor(&conv_descs[i]);
        
        cudnnSetTensor4dDescriptor(
            input_descs[i],
            CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT,
            batch_size,
            in_channels,
            height,
            width
        );
        
        cudnnSetFilter4dDescriptor(
            filter_descs[i],
            CUDNN_DATA_FLOAT,
            CUDNN_TENSOR_NCHW,
            filter_count,
            in_channels,
            filter_height,
            filter_width
        );
        
        cudnnSetConvolution2dDescriptor(
            conv_descs[i],
            1, 1, // padding
            1, 1, // stride
            1, 1, // dilation
            CUDNN_CROSS_CORRELATION,
            CUDNN_DATA_FLOAT
        );
        
        int out_n, out_c, out_h, out_w;
        cudnnGetConvolution2dForwardOutputDim(
            conv_descs[i],
            input_descs[i],
            filter_descs[i],
            &out_n,
            &out_c,
            &out_h,
            &out_w
        );
        
        cudnnSetTensor4dDescriptor(
            output_descs[i],
            CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT,
            out_n,
            out_c,
            out_h,
            out_w
        );
    }
    
    // Find best algorithm (version-gated: the legacy API was removed in cuDNN 9)
    cudnnConvolutionFwdAlgo_t algo;
#if CUDNN_MAJOR >= 8
    int algo_count = 0;
    cudnnConvolutionFwdAlgoPerf_t perf;
    cudnnGetConvolutionForwardAlgorithm_v7(
        handle,
        input_descs[0],
        filter_descs[0],
        conv_descs[0],
        output_descs[0],
        1,
        &algo_count,
        &perf
    );
    algo = (algo_count > 0) ? perf.algo : CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
#else
    cudnnGetConvolutionForwardAlgorithm(
        handle,
        input_descs[0],
        filter_descs[0],
        conv_descs[0],
        output_descs[0],
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
        0,
        &algo
    );
#endif
    
    // Get workspace size
    size_t workspace_size = 0;
    cudnnGetConvolutionForwardWorkspaceSize(
        handle,
        input_descs[0],
        filter_descs[0],
        conv_descs[0],
        output_descs[0],
        algo,
        &workspace_size
    );
    
    // Allocate workspace
    void* workspace = nullptr;
    if (workspace_size > 0) {
        cudaMalloc(&workspace, workspace_size);
    }
    
    // Perform batch convolution
    float alpha = 1.0f;
    float beta = 0.0f;
    
    for (int i = 0; i < batch_count; i++) {
        cudnnStatus_t status = cudnnConvolutionForward(
            handle,
            &alpha,
            input_descs[i],
            input_arrays[i],
            filter_descs[i],
            filter_arrays[i],
            conv_descs[i],
            algo,
            workspace,
            workspace_size,
            &beta,
            output_descs[i],
            output_arrays[i]
        );
        
        if (status != CUDNN_STATUS_SUCCESS) {
            // Cleanup
            if (workspace) cudaFree(workspace);
            for (int j = 0; j < batch_count; j++) {
                cudnnDestroyTensorDescriptor(input_descs[j]);
                cudnnDestroyTensorDescriptor(output_descs[j]);
                cudnnDestroyFilterDescriptor(filter_descs[j]);
                cudnnDestroyConvolutionDescriptor(conv_descs[j]);
            }
            delete[] input_descs;
            delete[] output_descs;
            delete[] filter_descs;
            delete[] conv_descs;
            return cudaErrorUnknown;
        }
    }
    
    // Cleanup
    if (workspace) cudaFree(workspace);
    for (int i = 0; i < batch_count; i++) {
        cudnnDestroyTensorDescriptor(input_descs[i]);
        cudnnDestroyTensorDescriptor(output_descs[i]);
        cudnnDestroyFilterDescriptor(filter_descs[i]);
        cudnnDestroyConvolutionDescriptor(conv_descs[i]);
    }
    delete[] input_descs;
    delete[] output_descs;
    delete[] filter_descs;
    delete[] conv_descs;
    
    if (stream == 0) {
        return cudaDeviceSynchronize();
    }
    
    return cudaSuccess;
}
