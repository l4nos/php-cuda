--TEST--
Multi-GPU device management
--SKIPIF--
<?php
if (!extension_loaded('cuda')) die('skip CUDA extension not loaded');
if (cuda_device_count() < 1) die('skip no CUDA device available');
?>
--INI--
display_errors=stderr
--FILE--
<?php
$count = cuda_device_count();
echo "Devices: $count\n";

// Every device must answer property queries
$ok = true;
for ($i = 0; $i < $count; $i++) {
    $props = cuda_device_properties($i);
    if (!isset($props['name'], $props['totalGlobalMem'])) {
        $ok = false;
    }
}
var_dump($ok);

// Device switching
var_dump(cuda_set_device(0));
var_dump(cuda_get_device() === 0);

if ($count >= 2) {
    var_dump(cuda_set_device(1));
    var_dump(cuda_get_device() === 1);

    // Tensors allocate on the current device
    $t = CudaTensor::ones([4, 4]);
    var_dump($t->device() === 1);

    var_dump(cuda_set_device(0));
    var_dump(cuda_get_device() === 0);
} else {
    echo "Single-GPU host: multi-device checks skipped\n";
}

// Invalid device must fail gracefully
var_dump(@cuda_set_device(9999) === false);

// Synchronization
var_dump(cuda_device_synchronize());
?>
--EXPECTF--
Devices: %d
bool(true)
bool(true)
bool(true)
%s
bool(true)
bool(true)
