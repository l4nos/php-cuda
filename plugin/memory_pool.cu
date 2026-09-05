#include "memory_pool.cuh"
#include <atomic>
#include <stdexcept>
#include <exception>

extern "C" cudaError_t cuda_memory_pool_create(
    MemoryPool** pool,
    size_t initial_size,
    cudaStream_t stream
) {
    if (initial_size < MIN_BLOCK_SIZE || initial_size > MAX_BLOCK_SIZE) {
        return cudaErrorInvalidValue;
    }

    try {
        *pool = new MemoryPool;
        (*pool)->blocks = new void*[MAX_POOL_BLOCKS];
        (*pool)->sizes = new size_t[MAX_POOL_BLOCKS];
        (*pool)->in_use = new bool[MAX_POOL_BLOCKS];
        (*pool)->block_locks = new std::atomic<bool>[MAX_POOL_BLOCKS];
        (*pool)->count = 0;
        (*pool)->total_size = 0;
        (*pool)->stream = stream;
        
        for (int i = 0; i < MAX_POOL_BLOCKS; i++) {
            (*pool)->blocks[i] = nullptr;
            (*pool)->sizes[i] = 0;
            (*pool)->in_use[i] = false;
            (*pool)->block_locks[i].store(false);
        }
        
        if (pthread_mutex_init(&(*pool)->mutex, NULL) != 0) {
            throw std::runtime_error("Mutex init failed");
        }
    } catch (const std::exception& e) {
        if (*pool) {
            delete[] (*pool)->blocks;
            delete[] (*pool)->sizes;
            delete[] (*pool)->in_use;
            delete[] (*pool)->block_locks;
            delete *pool;
        }
        return cudaErrorMemoryAllocation;
    }
    
    return cudaSuccess;
}

extern "C" cudaError_t cuda_memory_pool_allocate(
    MemoryPool* pool,
    size_t size,
    void** ptr
) {
    if (!pool || !ptr || size == 0 || size > MAX_BLOCK_SIZE) {
        return cudaErrorInvalidValue;
    }

    pthread_mutex_lock(&pool->mutex);
    
    // First try to find an existing free block
    for (int i = 0; i < pool->count; i++) {
        bool expected = false;
        if (!pool->in_use[i] && pool->sizes[i] >= size &&
            pool->block_locks[i].compare_exchange_strong(expected, true)) {
            *ptr = pool->blocks[i];
            pool->in_use[i] = true;
            pool->block_locks[i].store(false);
            pthread_mutex_unlock(&pool->mutex);
            return cudaSuccess;
        }
    }
    
    // Need to allocate new block
    if (pool->count >= MAX_POOL_BLOCKS) {
        pthread_mutex_unlock(&pool->mutex);
        return cudaErrorMemoryAllocation;
    }
    
    cudaError_t err = cudaMalloc(ptr, size);
    if (err != cudaSuccess) {
        pthread_mutex_unlock(&pool->mutex);
        return err;
    }
    
    pool->blocks[pool->count] = *ptr;
    pool->sizes[pool->count] = size;
    pool->in_use[pool->count] = true;
    pool->block_locks[pool->count].store(false);
    pool->count++;
    pool->total_size += size;
    
    pthread_mutex_unlock(&pool->mutex);
    return cudaSuccess;
}

extern "C" cudaError_t cuda_memory_pool_free(
    MemoryPool* pool,
    void* ptr
) {
    if (!pool || !ptr) {
        return cudaErrorInvalidValue;
    }

    pthread_mutex_lock(&pool->mutex);
    
    for (int i = 0; i < pool->count; i++) {
        if (pool->blocks[i] == ptr) {
            bool expected = false;
            if (pool->block_locks[i].compare_exchange_strong(expected, true)) {
                pool->in_use[i] = false;
                pool->block_locks[i].store(false);
                pthread_mutex_unlock(&pool->mutex);
                return cudaSuccess;
            }
        }
    }
    
    pthread_mutex_unlock(&pool->mutex);
    return cudaErrorInvalidValue;
}

extern "C" cudaError_t cuda_memory_pool_destroy(
    MemoryPool* pool
) {
    if (!pool) {
        return cudaErrorInvalidValue;
    }

    pthread_mutex_lock(&pool->mutex);
    
    for (int i = 0; i < pool->count; i++) {
        if (pool->blocks[i]) {
            cudaFree(pool->blocks[i]);
        }
    }
    
    delete[] pool->blocks;
    delete[] pool->sizes;
    delete[] pool->in_use;
    delete[] pool->block_locks;
    
    pthread_mutex_unlock(&pool->mutex);
    pthread_mutex_destroy(&pool->mutex);
    delete pool;
    
    return cudaSuccess;
}

extern "C" cudaError_t cuda_memory_pool_resize(MemoryPool* pool) {
    // Implementation for dynamic resizing
    return cudaSuccess;
}

extern "C" cudaError_t cuda_memory_pool_defragment(MemoryPool* pool) {
    // Implementation for defragmentation
    return cudaSuccess;
}

extern "C" cudaError_t cuda_memory_pool_get_stats(
    MemoryPool* pool,
    size_t* free,
    size_t* total
) {
    if (!pool || !free || !total) {
        return cudaErrorInvalidValue;
    }

    pthread_mutex_lock(&pool->mutex);
    
    *total = pool->total_size;
    *free = 0;
    
    for (int i = 0; i < pool->count; i++) {
        if (!pool->in_use[i]) {
            *free += pool->sizes[i];
        }
    }
    
    pthread_mutex_unlock(&pool->mutex);
    return cudaSuccess;
}
