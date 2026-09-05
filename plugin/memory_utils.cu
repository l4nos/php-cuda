#include "memory_utils.cuh"

extern "C" cudaError_t cuda_pinned_malloc(void** ptr, size_t size) {
    return cudaHostAlloc(ptr, size, cudaHostAllocDefault);
}

extern "C" cudaError_t cuda_pinned_free(void* ptr) {
    return cudaFreeHost(ptr);
}

extern "C" cudaError_t cuda_unified_malloc(
    void** ptr,
    size_t size,
    MemoryConfig* config
) {
    cudaError_t err = cudaMallocManaged(ptr, size);
    if (err != cudaSuccess) return err;

    // Apply memory hints
    if (config->read_mostly) {
        err = cudaMemAdvise(*ptr, size, cudaMemAdviseSetReadMostly, 0);
        if (err != cudaSuccess) return err;
    }

    if (config->preferred_location >= 0) {
        err = cudaMemAdvise(*ptr, size, cudaMemAdviseSetPreferredLocation, 
                           config->preferred_location);
        if (err != cudaSuccess) return err;
    }

    return cudaSuccess;
}

extern "C" cudaError_t cuda_unified_free(void* ptr) {
    return cudaFree(ptr);
}

extern "C" cudaError_t cuda_unified_prefetch(
    void* ptr,
    size_t size,
    int device
) {
    cudaStream_t stream;
    cudaError_t err = cudaStreamCreate(&stream);
    if (err != cudaSuccess) return err;

    err = cudaMemPrefetchAsync(ptr, size, device, stream);
    if (err != cudaSuccess) {
        cudaStreamDestroy(stream);
        return err;
    }

    err = cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    return err;
}

extern "C" cudaError_t cuda_set_memory_hint(
    void* ptr,
    size_t size,
    cudaMemoryAdvise advice
) {
    return cudaMemAdvise(ptr, size, advice, 0);
}

extern "C" cudaError_t cuda_optimize_memory_access(
    void* ptr,
    size_t size,
    int device
) {
    // First, set preferred location
    cudaError_t err = cudaMemAdvise(ptr, size, 
                                   cudaMemAdviseSetPreferredLocation, device);
    if (err != cudaSuccess) return err;

    // Then, set accessed-by hint
    err = cudaMemAdvise(ptr, size, cudaMemAdviseSetAccessedBy, device);
    if (err != cudaSuccess) return err;

    // Finally, prefetch to device
    return cuda_unified_prefetch(ptr, size, device);
}

extern "C" cudaError_t cuda_measure_memory_bandwidth(
    size_t size,
    float* bandwidth
) {
    void *d_a, *d_b;
    cudaEvent_t start, stop;
    float elapsed_time;

    // Allocate memory
    cudaError_t err = cudaMalloc(&d_a, size);
    if (err != cudaSuccess) return err;

    err = cudaMalloc(&d_b, size);
    if (err != cudaSuccess) {
        cudaFree(d_a);
        return err;
    }

    // Create events
    err = cudaEventCreate(&start);
    if (err != cudaSuccess) {
        cudaFree(d_a);
        cudaFree(d_b);
        return err;
    }

    err = cudaEventCreate(&stop);
    if (err != cudaSuccess) {
        cudaEventDestroy(start);
        cudaFree(d_a);
        cudaFree(d_b);
        return err;
    }

    // Measure bandwidth
    cudaEventRecord(start, 0);
    err = cudaMemcpy(d_b, d_a, size, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_a);
        cudaFree(d_b);
        return err;
    }
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);

    // Calculate bandwidth in GB/s
    *bandwidth = (size / (1024.0f * 1024.0f * 1024.0f)) / (elapsed_time / 1000.0f);

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a);
    cudaFree(d_b);

    return cudaSuccess;
}
