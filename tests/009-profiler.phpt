--TEST--
Profiling: events, memory info, profiler control
--SKIPIF--
<?php
if (!extension_loaded('cuda')) die('skip CUDA extension not loaded');
if (cuda_device_count() < 1) die('skip no CUDA device available');
?>
--INI--
display_errors=stderr
--FILE--
<?php
// Event timing around a real operation
function test_event_timing() {
    $event = cuda_event_create();
    if ($event === false) return false;

    cuda_event_record_start($event);

    $a = CudaTensor::rand([512, 512]);
    $b = CudaTensor::rand([512, 512]);
    $c = $a->matmul($b);

    cuda_event_record_stop($event);

    $ms = cuda_event_elapsed_time($event);
    echo "Matmul took: " . round($ms, 3) . " ms\n";

    cuda_event_destroy($event);
    return $ms >= 0.0;
}

// Profiler start/stop control
function test_profiler_control() {
    if (!cuda_profiler_start()) return false;
    $t = CudaTensor::ones([8, 8])->relu();
    if (!cuda_profiler_stop()) return false;
    return true;
}

// Memory info shape
function test_memory_info() {
    $info = cuda_memory_get_info();
    return isset($info['free'], $info['total'], $info['used'])
        && $info['total'] > 0
        && $info['used'] === $info['total'] - $info['free'];
}

var_dump(test_event_timing());
var_dump(test_profiler_control());
var_dump(test_memory_info());
?>
--EXPECTF--
Matmul took: %f ms
bool(true)
bool(true)
bool(true)
