--TEST--
NVRTC runtime kernel compilation and launch
--SKIPIF--
<?php
if (!extension_loaded('cuda')) die('skip CUDA extension not loaded');
if (cuda_device_count() < 1) die('skip no CUDA device available');
if (!function_exists('cuda_kernel_compile')) die('skip built without NVRTC');
?>
--INI--
display_errors=stderr
--FILE--
<?php
// saxpy: y = a*x + y
// NOTE: PHP floats are passed to kernels as C doubles (8 bytes); scalar
// float parameters in custom kernels must be declared `double`.
$source = '
extern "C" __global__ void saxpy(double a, const float* x, float* y, long long n) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = (float)a * x[idx] + y[idx];
    }
}
';

$kernel = cuda_kernel_compile($source, 'saxpy');
var_dump($kernel !== false);

$n = 1024;
$x = CudaTensor::ones([$n]);
$y = CudaTensor::full([$n], 2.0);

$grid = [intdiv($n + 255, 256)];
$block = [256];

// y = 3.0 * 1.0 + 2.0 = 5.0
var_dump(cuda_kernel_launch($kernel, [3.0, $x, $y, $n], $grid, $block));
cuda_device_synchronize();

$vals = $y->toArray();
$ok = true;
foreach ($vals as $v) {
    if (abs($v - 5.0) > 1e-5) { $ok = false; break; }
}
var_dump($ok);

// Compilation errors must throw CudaException with the NVRTC log
try {
    cuda_kernel_compile('this is not cuda', 'nope');
    var_dump(false);
} catch (CudaException $e) {
    var_dump(true);
}
?>
--EXPECT--
bool(true)
bool(true)
bool(true)
bool(true)
