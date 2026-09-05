--TEST--
CudaTensor: creation, ops, views, dtypes
--SKIPIF--
<?php
if (!extension_loaded('cuda')) die('skip CUDA extension not loaded');
if (cuda_device_count() < 1) die('skip no CUDA device available');
?>
--INI--
display_errors=stderr
--FILE--
<?php
function approx($a, $b, $eps = 1e-4) { return abs($a - $b) < $eps; }

// Creation + introspection
$t = CudaTensor::fromArray([[1.0, 2.0], [3.0, 4.0]]);
var_dump($t->shape());
var_dump($t->ndim());
var_dump($t->size());
var_dump($t->dtype() === CudaTensor::FP32);
var_dump($t->device() === 0);
echo $t, "\n";

// toArray round trip
$back = $t->toArray();
var_dump(approx($back[0][0], 1.0) && approx($back[1][1], 4.0));

// Elementwise with scalar
$r = $t->add(1.0)->toArray();
var_dump(approx($r[0][0], 2.0) && approx($r[1][1], 5.0));

// Elementwise tensor-tensor
$u = CudaTensor::ones([2, 2]);
$r = $t->add($u)->toArray();
var_dump(approx($r[0][1], 3.0));

// Broadcasting: [2,2] + [2]
$v = CudaTensor::fromArray([10.0, 20.0]);
$r = $t->add($v)->toArray();
var_dump(approx($r[0][0], 11.0) && approx($r[0][1], 22.0) && approx($r[1][0], 13.0));

// mul / sub / div
$r = $t->mul(2.0)->sub(1.0)->div(2.0)->toArray();
var_dump(approx($r[0][0], 0.5) && approx($r[1][1], 3.5));

// In-place
$w = CudaTensor::zeros([2, 2]);
$w->add_(5.0)->mul_(2.0);
var_dump(approx($w->toArray()[0][0], 10.0));

// matmul numeric check
$a = CudaTensor::fromArray([[1.0, 2.0], [3.0, 4.0]]);
$b = CudaTensor::fromArray([[5.0, 6.0], [7.0, 8.0]]);
$c = $a->matmul($b)->toArray();
var_dump(approx($c[0][0], 19.0) && approx($c[0][1], 22.0) && approx($c[1][0], 43.0) && approx($c[1][1], 50.0));

// Activations
$r = CudaTensor::fromArray([-1.0, 0.0, 1.0])->relu()->toArray();
var_dump(approx($r[0], 0.0) && approx($r[2], 1.0));
$r = CudaTensor::fromArray([0.0])->sigmoid()->toArray();
var_dump(approx($r[0], 0.5));

// softmax over last dim: rows sum to 1
$s = CudaTensor::fromArray([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])->softmax()->toArray();
var_dump(approx($s[0][0] + $s[0][1] + $s[0][2], 1.0));
var_dump(approx($s[1][0], 1.0 / 3.0, 1e-3));

// Reductions
$t2 = CudaTensor::fromArray([[1.0, 2.0], [3.0, 4.0]]);
var_dump(approx($t2->sum(), 10.0));
var_dump(approx($t2->mean(), 2.5));
var_dump(approx($t2->max(), 4.0));
var_dump(approx($t2->min(), 1.0));

// Views: reshape / transpose / slice share storage
$base = CudaTensor::fromArray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
$flat = $base->reshape([6]);
var_dump($flat->shape() === [6]);
$tr = $base->transpose();
var_dump($tr->shape() === [3, 2]);
$trd = $tr->toArray();
var_dump(approx($trd[0][0], 1.0) && approx($trd[0][1], 4.0) && approx($trd[2][1], 6.0));
$sl = $base->slice(0, 1, 1);
var_dump($sl->shape() === [1, 3]);
var_dump(approx($sl->toArray()[0][0], 4.0));

// contiguous() materializes a view
$ct = $tr->contiguous();
var_dump($ct->shape() === [3, 2]);

// dtypes
$f64 = CudaTensor::fromArray([1.5, 2.5], CudaTensor::FP64);
var_dump($f64->dtype() === CudaTensor::FP64);
var_dump(approx($f64->sum(), 4.0, 1e-9));
$i32 = CudaTensor::fromArray([1, 2, 3], CudaTensor::INT32);
var_dump($i32->dtype() === CudaTensor::INT32);
var_dump(approx($i32->sum(), 6.0));
$f16 = CudaTensor::fromArray([1.0, 2.0], CudaTensor::FP16);
var_dump(approx($f16->sum(), 3.0, 1e-2));

// Broadcast incompatibility throws
try {
    CudaTensor::ones([2, 3])->add(CudaTensor::ones([4]));
    var_dump(false);
} catch (CudaException $e) {
    var_dump(true);
}
?>
--EXPECT--
array(2) {
  [0]=>
  int(2)
  [1]=>
  int(2)
}
int(2)
int(4)
bool(true)
bool(true)
CudaTensor(shape=[2, 2], dtype=fp32, device=0)
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
bool(true)
