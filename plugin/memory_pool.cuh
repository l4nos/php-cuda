#ifndef MEMORY_POOL_CUH
#define MEMORY_POOL_CUH

#include <cuda_runtime.h>
#include <pthread.h>
#include <atomic>
#include "cuda_utils.cuh"

#define MAX_POOL_BLOCKS 1024  // Increased from 16
#define MIN_BLOCK_SIZE 256    // Minimum allocation size
#define MAX_BLOCK_SIZE (1024 * 1024 * 1024)  // 1GB max allocation

// Memory Pool Management
struct MemoryPool {
    void** blocks;
    size_t* sizes;
    bool* in_use;
    int count;
    size_t total_size;
    cudaStream_t stream;
    pthread_mutex_t mutex;
    std::atomic<bool>* block_locks;  // Added for atomic operations
};

extern "C" {
    cudaError_t cuda_memory_pool_create(MemoryPool** pool, size_t initial_size, cudaStream_t stream);
    cudaError_t cuda_memory_pool_allocate(MemoryPool* pool, size_t size, void** ptr);
    cudaError_t cuda_memory_pool_free(MemoryPool* pool, void* ptr);
    cudaError_t cuda_memory_pool_destroy(MemoryPool* pool);
    
    // Added functions for better management
    cudaError_t cuda_memory_pool_resize(MemoryPool* pool);
    cudaError_t cuda_memory_pool_defragment(MemoryPool* pool);
    cudaError_t cuda_memory_pool_get_stats(MemoryPool* pool, size_t* free, size_t* total);
}

#endif // MEMORY_POOL_CUH
