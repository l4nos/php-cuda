--TEST--
Advanced memory: pinned, unified, bandwidth
--SKIPIF--
<?php
if (!extension_loaded('cuda')) die('skip CUDA extension not loaded');
if (cuda_device_count() < 1) die('skip no CUDA device available');
?>
--INI--
display_errors=stderr
--FILE--
<?php
// Pinned memory round trip
function test_pinned() {
    $mem = cuda_pinned_alloc(64);
    if ($mem === false) {
        echo "Pinned allocation failed\n";
        return false;
    }
    $data = str_repeat("\xAB", 64);
    if (!cuda_memcpy_host_to_device($mem, $data)) return false;
    $back = cuda_memcpy_device_to_host($mem, 64);
    cuda_free($mem);
    return $back === $data;
}

// Unified memory round trip
function test_unified() {
    $mem = cuda_unified_alloc(64);
    if ($mem === false) {
        echo "Unified allocation failed\n";
        return false;
    }
    $data = pack('f16', ...array_map(fn($i) => (float)$i, range(1, 16)));
    if (!cuda_memcpy_host_to_device($mem, $data)) return false;
    $back = cuda_memcpy_device_to_host($mem, 64);
    cuda_free($mem);
    return $back === $data;
}

// Bandwidth measurement returns a sane positive number
function test_bandwidth() {
    $bw = cuda_measure_memory_bandwidth(16 * 1024 * 1024);
    if ($bw === false || $bw <= 0.0) {
        echo "Bandwidth measurement failed\n";
        return false;
    }
    echo "Bandwidth: " . round($bw, 1) . " GB/s\n";
    return true;
}

var_dump(test_pinned());
var_dump(test_unified());
var_dump(test_bandwidth());
?>
--EXPECTF--
bool(true)
bool(true)
Bandwidth: %f GB/s
bool(true)
