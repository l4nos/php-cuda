--TEST--
Memory management, pools and leak detection
--SKIPIF--
<?php
if (!extension_loaded('cuda')) die('skip CUDA extension not loaded');
if (cuda_device_count() < 1) die('skip no CUDA device available');
?>
--INI--
display_errors=stderr
--FILE--
<?php
// Test memory allocation and deallocation
function test_memory_management() {
    $sizes = [1024, 2048, 4096, 8192];
    $ptrs = [];

    foreach ($sizes as $size) {
        $ptr = cuda_malloc($size);
        if ($ptr === false) {
            echo "Failed to allocate {$size} bytes\n";
            return false;
        }
        $ptrs[] = $ptr;
    }

    $info = cuda_memory_get_info();
    if (!isset($info['free'], $info['total'], $info['used'])) {
        echo "cuda_memory_get_info returned unexpected shape\n";
        return false;
    }
    $free_before = $info['free'];

    foreach ($ptrs as $ptr) {
        if (!cuda_free($ptr)) {
            echo "Failed to free memory\n";
            return false;
        }
    }

    $free_after = cuda_memory_get_info()['free'];
    if ($free_after < $free_before) {
        echo "Possible memory leak detected\n";
        return false;
    }

    return true;
}

// Test memset + memcpy round trip
function test_memcpy() {
    $mem = cuda_malloc(16);
    if ($mem === false) return false;

    if (!cuda_memset($mem, 0)) return false;

    $data = pack('f4', 1.0, 2.0, 3.0, 4.0);
    if (!cuda_memcpy_host_to_device($mem, $data)) return false;

    $back = cuda_memcpy_device_to_host($mem, 16);
    if ($back !== $data) {
        echo "memcpy round trip mismatch\n";
        return false;
    }

    // device-to-device
    $mem2 = cuda_malloc(16);
    if (!cuda_memcpy_device_to_device($mem2, $mem, 16)) return false;
    $back2 = cuda_memcpy_device_to_host($mem2, 16);
    if ($back2 !== $data) {
        echo "device-to-device copy mismatch\n";
        return false;
    }

    cuda_free($mem);
    cuda_free($mem2);
    return true;
}

// Test memory pool (grows on demand; freed blocks are reused)
function test_memory_pool() {
    $pool = cuda_memory_pool_init(1024 * 1024);
    if ($pool === false) {
        echo "Failed to initialize memory pool\n";
        return false;
    }

    $ptrs = [];
    for ($i = 0; $i < 10; $i++) {
        $ptr = cuda_memory_pool_allocate($pool, 1024);
        if ($ptr === false) {
            echo "Failed to allocate from pool\n";
            return false;
        }
        $ptrs[] = $ptr;
    }

    $stats = cuda_memory_pool_stats($pool);
    echo "Pool total size: {$stats['total_size']}\n";
    echo "Pool used size: {$stats['used_size']}\n";

    foreach (array_slice($ptrs, 0, 5) as $ptr) {
        if (!cuda_memory_pool_free($pool, $ptr)) {
            echo "Failed to free memory to pool\n";
            return false;
        }
    }

    $stats = cuda_memory_pool_stats($pool);
    echo "Pool used size after partial free: {$stats['used_size']}\n";

    // Allocating again should reuse freed blocks (total must not grow)
    for ($i = 0; $i < 5; $i++) {
        $ptrs[] = cuda_memory_pool_allocate($pool, 1024);
    }
    $stats = cuda_memory_pool_stats($pool);
    echo "Pool total size after reuse: {$stats['total_size']}\n";

    if (!cuda_memory_pool_destroy($pool)) {
        echo "Failed to destroy memory pool\n";
        return false;
    }

    return true;
}

var_dump(test_memory_management());
var_dump(test_memcpy());
var_dump(test_memory_pool());
?>
--EXPECT--
bool(true)
bool(true)
Pool total size: 10240
Pool used size: 10240
Pool used size after partial free: 5120
Pool total size after reuse: 10240
bool(true)
