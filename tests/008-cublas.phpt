--TEST--
cuBLAS operations and performance
--SKIPIF--
<?php
if (!extension_loaded('cuda')) die('skip CUDA extension not loaded');
if (cuda_device_count() < 1) die('skip no CUDA device available');
?>
--INI--
display_errors=stderr
--FILE--
<?php
function approx($a, $b, $eps = 1e-2) { return abs($a - $b) < $eps; }

// Basic cuBLAS correctness (non-square, verifies row-major handling)
function test_cublas_basic() {
    $handle = cuda_cublas_create();
    if ($handle === false) return false;

    $m = 2; $n = 3; $k = 4;
    // A (m x k) row-major flat
    $a = [1, 2, 3, 4,
          5, 6, 7, 8];
    // B (k x n) row-major flat
    $b = [1, 2, 3,
          4, 5, 6,
          7, 8, 9,
          10, 11, 12];

    $c = [];
    if (!cuda_cublas_matrix_multiply($handle, $a, $b, $c, $m, $n, $k)) {
        echo "cuBLAS matrix multiply failed\n";
        return false;
    }

    // Expected C (m x n):
    // row0: 1*1+2*4+3*7+4*10=70, 1*2+2*5+3*8+4*11=80, 1*3+2*6+3*9+4*12=90
    // row1: 5*1+6*4+7*7+8*10=158, 5*2+6*5+7*8+8*11=184, 5*3+6*6+7*9+8*12=210
    $expected = [70, 80, 90, 158, 184, 210];
    foreach ($expected as $i => $e) {
        if (!approx($c[$i], $e)) {
            echo "Mismatch at $i: got {$c[$i]}, expected $e\n";
            return false;
        }
    }

    // GEMM with alpha/beta: C = 2*A*B + 1*C
    $c2 = $c; // initial C for beta
    if (!cuda_cublas_gemm($handle, $a, $b, $c2, $m, $n, $k, 2.0, 1.0)) {
        echo "cuBLAS gemm failed\n";
        return false;
    }
    foreach ($expected as $i => $e) {
        if (!approx($c2[$i], 3.0 * $e)) {
            echo "GEMM mismatch at $i: got {$c2[$i]}, expected " . (3.0 * $e) . "\n";
            return false;
        }
    }

    cuda_cublas_destroy($handle);
    return true;
}

// Batched GEMM
function test_cublas_batch() {
    $handle = cuda_cublas_create();

    $batch = 8;
    $size = 4;
    $a = array_fill(0, $batch, array_fill(0, $size * $size, 1.0));
    $b = array_fill(0, $batch, array_fill(0, $size * $size, 2.0));
    $results = [];

    if (!cuda_batch_gemm($handle, $a, $b, $results, $size, $size, $size, $batch)) {
        echo "Batch GEMM failed\n";
        return false;
    }

    if (count($results) !== $batch) return false;
    // Each element: sum of 4 products of 1.0*2.0 = 8.0
    foreach ($results as $mat) {
        foreach ($mat as $v) {
            if (!approx($v, 8.0)) return false;
        }
    }

    cuda_cublas_destroy($handle);
    return true;
}

// Performance smoke test
function test_cublas_performance() {
    $handle = cuda_cublas_create();
    $sizes = [128, 512, 1024];
    $iterations = 5;

    foreach ($sizes as $size) {
        $a = array_fill(0, $size * $size, 1.0);
        $b = array_fill(0, $size * $size, 2.0);
        $c = [];

        $start = microtime(true);
        for ($i = 0; $i < $iterations; $i++) {
            if (!cuda_cublas_matrix_multiply($handle, $a, $b, $c, $size, $size, $size)) {
                echo "Perf test failed at size $size\n";
                return false;
            }
        }
        $duration = microtime(true) - $start;
        $gflops = (2.0 * $size * $size * $size * $iterations) / ($duration * 1e9);
        echo "Size {$size}x{$size}: " . round($gflops, 1) . " GFLOPS\n";
    }

    cuda_cublas_destroy($handle);
    return true;
}

var_dump(test_cublas_basic());
var_dump(test_cublas_batch());
var_dump(test_cublas_performance());
?>
--EXPECTF--
bool(true)
bool(true)
Size 128x128: %f GFLOPS
Size 512x512: %f GFLOPS
Size 1024x1024: %f GFLOPS
bool(true)
