--TEST--
Streams, events and CUDA graphs
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
// Stream lifecycle
$stream = cuda_stream_create();
var_dump($stream !== false);
var_dump(cuda_stream_synchronize($stream));
var_dump(cuda_stream_query($stream));

// Event timing on a stream
$event = cuda_event_create();
cuda_event_record_start($event, $stream);
$t = CudaTensor::rand([256, 256])->matmul(CudaTensor::rand([256, 256]));
cuda_event_record_stop($event, $stream);
$ms = cuda_event_elapsed_time($event);
var_dump($ms >= 0.0);

// Graph capture/replay of an NVRTC kernel on a captured stream
$source = '
extern "C" __global__ void add_one(float* x, long long n) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) x[idx] += 1.0f;
}
';
$kernel = cuda_kernel_compile($source, 'add_one');

$n = 512;
$x = CudaTensor::zeros([$n]);

var_dump(cuda_graph_begin_capture($stream));
var_dump(cuda_kernel_launch($kernel, [$x, $n], [2], [256], $stream));
$graph = cuda_graph_end_capture();
var_dump($graph !== false);

// Replay the graph 3 times: x should become 3.0
for ($i = 0; $i < 3; $i++) {
    var_dump(cuda_graph_launch($graph, $stream));
}
cuda_stream_synchronize($stream);

$v = $x->toArray();
var_dump(abs($v[0] - 3.0) < 1e-5 && abs($v[$n - 1] - 3.0) < 1e-5);

var_dump(cuda_graph_destroy($graph));
var_dump(cuda_event_destroy($event));
var_dump(cuda_stream_destroy($stream));
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
bool(true)
bool(true)
bool(true)
bool(true)
bool(true)
bool(true)
