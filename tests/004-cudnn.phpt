--TEST--
cuDNN convolution forward
--SKIPIF--
<?php
if (!extension_loaded('cuda')) die('skip CUDA extension not loaded');
if (cuda_device_count() < 1) die('skip no CUDA device available');
if (!function_exists('cuda_cudnn_convolution_forward')) die('skip built without cuDNN');
?>
--INI--
display_errors=stderr
--FILE--
<?php
// 1x1x4x4 input, 1x1x3x3 filter, stride 1, padding 0 -> 1x1x2x2 output
$input = [
    1.0, 2.0, 3.0, 4.0,
    5.0, 6.0, 7.0, 8.0,
    9.0, 10.0, 11.0, 12.0,
    13.0, 14.0, 15.0, 16.0,
];
$filter = array_fill(0, 9, 1.0); // 3x3 of ones -> sums each 3x3 window

$output = [];
$ok = cuda_cudnn_convolution_forward(
    $input, $filter, $output,
    1, 1, 4, 4,  // batch, channels, h, w
    1, 3, 3,     // filters, fh, fw
    1, 0         // stride, padding
);
var_dump($ok);
var_dump($output['shape']);

// Expected window sums:
// [1+2+3+5+6+7+9+10+11]      = 54
// [2+3+4+6+7+8+10+11+12]     = 63
// [5+6+7+9+10+11+13+14+15]   = 90
// [6+7+8+10+11+12+14+15+16]  = 99
$expected = [54.0, 63.0, 90.0, 99.0];
foreach ($expected as $i => $e) {
    if (abs($output['data'][$i] - $e) > 1e-4) {
        echo "Mismatch at $i: got {$output['data'][$i]}, expected $e\n";
    }
}
var_dump(true);
?>
--EXPECT--
bool(true)
array(4) {
  [0]=>
  int(1)
  [1]=>
  int(1)
  [2]=>
  int(2)
  [3]=>
  int(2)
}
bool(true)
