# Installing on Ubuntu 24.04

This is the reference platform. Other Debian/Ubuntu versions work the same way;
adjust package names for your PHP version.

## 1. Install the NVIDIA driver

Use Ubuntu's driver packages or NVIDIA's repository. Verify with:

```bash
nvidia-smi
```

Note the "CUDA Version" in the top-right corner: that is the **maximum CUDA
runtime your driver supports**. Your toolkit (below) must not exceed it, or you
need the `cuda-compat` package or a newer driver.

## 2. Install the CUDA toolkit (NVIDIA repository)

Do **not** install Ubuntu's `nvidia-cuda-toolkit` — it conflicts with the
NVIDIA repo packages and is a common source of broken, mixed toolchains.

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-6
```

Add to your shell profile:

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

Verify the toolchain is consistent — both should report the same major.minor:

```bash
nvcc --version
cat /usr/local/cuda/version.json
```

## 3. Install cuDNN 9

```bash
sudo apt-get install -y libcudnn9-dev-cuda-12
```

## 4. Install PHP development files

```bash
sudo apt-get install -y php-cli php-dev   # or php8.4-dev from ppa:ondrej/php
```

## 5. Build and install the extension

```bash
git clone https://github.com/l4nos/php-cuda.git
cd php-cuda/plugin
./compile.sh
```

The script auto-detects the toolkit, checks driver/runtime compatibility,
builds, installs, and writes `cuda.ini`. Useful overrides:

```bash
CUDA_ARCH="75" ./compile.sh        # build only for your GPU (faster)
CUDNN_PATH=no ./compile.sh         # build without cuDNN
ENABLE_NVTX=1 ./compile.sh         # enable NVTX profiling ranges
./compile.sh --test                # build + run the test suite
./compile.sh --uninstall           # remove
```

## 6. Verify

```bash
php -m | grep cuda
php -r 'var_dump(cuda_device_count());'
php -r '$t = CudaTensor::ones([2,2]); var_dump($t->add(1.0)->toArray());'
```

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Unsupported gpu architecture 'compute_30'` | Old build files referencing sm_30 | Re-clone or `git pull`; sm_30 was removed in CUDA 12 |
| `cudnnGetConvolutionForwardAlgorithm` not found | cuDNN 9 removed the legacy API | Update to the current source (version-gated) |
| `nvToolsExt.h` not found / `-lnvToolsExt` fails | CUDA 12 NVTX is header-only (`nvtx3/`) | Handled automatically; NVTX is optional |
| Extension loads but `cuda_device_count()` errors | Driver older than toolkit runtime | `nvidia-smi` CUDA Version must be >= toolkit version; upgrade driver or install `cuda-compat-12-x` |
| `nvcc` 12.0 but toolkit 12.x installed | Mixed toolchains on PATH | `compile.sh` refuses this; fix `PATH`/`CUDA_PATH` |
