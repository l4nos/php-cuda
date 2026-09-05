--TEST--
Stress testing: allocation churn, tensor churn, error recovery
--SKIPIF--
<?php
if (!extension_loaded('cuda')) die('skip CUDA extension not loaded');
if (cuda_device_count() < 1) die('skip no CUDA device available');
?>
--INI--
display_errors=stderr
--FILE--
<?php
// Allocation churn: many alloc/free cycles must not leak
function test_alloc_churn() {
    $free_start = cuda_memory_get_info()['free'];
    for ($i = 0; $i < 200; $i++) {
        $ptr = cuda_malloc(1024 * 1024); // 1MB
        if ($ptr === false) {
            echo "Allocation failed at iteration $i\n";
            return false;
        }
        cuda_free($ptr);
    }
    $free_end = cuda_memory_get_info()['free'];
    // Allow small driver-side fragmentation slack (16MB)
    if ($free_start - $free_end > 16 * 1024 * 1024) {
        echo "Leak suspected: lost " . ($free_start - $free_end) . " bytes\n";
        return false;
    }
    return true;
}

// Tensor churn: objects going out of scope must free device memory
function test_tensor_churn() {
    $free_start = cuda_memory_get_info()['free'];
    for ($i = 0; $i < 100; $i++) {
        $t = CudaTensor::rand([256, 256]);
        $u = $t->add(1.0)->relu();
        unset($t, $u);
    }
    gc_collect_cycles();
    $free_end = cuda_memory_get_info()['free'];
    if ($free_start - $free_end > 16 * 1024 * 1024) {
        echo "Tensor leak suspected\n";
        return false;
    }
    return true;
}

// Error recovery: failed operations must not poison subsequent ones
function test_error_recovery() {
    $a = CudaTensor::ones([4, 4]);
    $b = CudaTensor::ones([8, 8]);

    // Broadcast-incompatible: must throw CudaException or warn+false path
    try {
        $a->add($b);
        echo "Expected broadcast failure did not occur\n";
        return false;
    } catch (CudaException $e) {
        // expected
    }

    // Extension must still work afterwards
    $c = $a->add(2.0);
    $vals = $c->toArray();
    foreach ($vals as $row) {
        foreach ($row as $v) {
            if (abs($v - 3.0) > 1e-5) return false;
        }
    }
    return true;
}

// Sustained compute: repeated matmuls
function test_sustained_compute() {
    $a = CudaTensor::full([64, 64], 0.5);
    $b = CudaTensor::full([64, 64], 2.0);
    for ($i = 0; $i < 50; $i++) {
        $c = $a->matmul($b);
        unset($c);
    }
    return true;
}

var_dump(test_alloc_churn());
var_dump(test_tensor_churn());
var_dump(test_error_recovery());
var_dump(test_sustained_compute());
?>
--EXPECT--
bool(true)
bool(true)
bool(true)
bool(true)
