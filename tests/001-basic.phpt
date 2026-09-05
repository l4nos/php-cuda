--TEST--
Basic CUDA functionality test
--SKIPIF--
<?php
if (!extension_loaded('cuda')) die('skip CUDA extension not loaded');
if (cuda_device_count() < 1) die('skip no CUDA device available');
?>
--INI--
display_errors=stderr
--FILE--
<?php
// Test device count
$count = cuda_device_count();
var_dump($count > 0);

// Test device properties
$props = cuda_device_properties(0);
var_dump(isset($props['name']));
var_dump(isset($props['totalGlobalMem']));
var_dump(isset($props['maxThreadsPerBlock']));
var_dump(isset($props['computeCapabilityMajor']));

// Test device get/set
var_dump(cuda_get_device() === 0);
var_dump(cuda_set_device(0));

// Test matrix multiplication (non-square to exercise dimension handling)
$matrix_a = [
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
]; // 2x3

$matrix_b = [
    [7.0, 8.0],
    [9.0, 10.0],
    [11.0, 12.0],
]; // 3x2

$result = [];
$success = cuda_matrix_multiply($matrix_a, $matrix_b, $result);
var_dump($success);

// Expected:
// [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
// [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
var_dump($result);

// Test error handling
$last_error = cuda_get_last_error();
var_dump($last_error === 0); // 0 means cudaSuccess

// Inner dimension mismatch must fail gracefully
$bad_b = [[1.0, 2.0]]; // 1x2, needs 3 rows
$result2 = [];
$success2 = @cuda_matrix_multiply($matrix_a, $bad_b, $result2);
var_dump($success2 === false);

// Ragged matrix must fail gracefully
$ragged = [[1.0, 2.0, 3.0], [4.0]];
$result3 = [];
$success3 = @cuda_matrix_multiply($ragged, $matrix_b, $result3);
var_dump($success3 === false);

// Error string helpers
var_dump(is_string(cuda_get_error_string(0)));
var_dump(is_string(cuda_get_error_name(0)));
?>
--EXPECT--
bool(true)
bool(true)
bool(true)
bool(true)
bool(true)
bool(true)
bool(true)
bool(true)
array(2) {
  [0]=>
  array(2) {
    [0]=>
    float(58)
    [1]=>
    float(64)
  }
  [1]=>
  array(2) {
    [0]=>
    float(139)
    [1]=>
    float(154)
  }
}
bool(true)
bool(true)
bool(true)
bool(true)
bool(true)
