#include "profiler.cuh"

/*
 * Intentionally minimal: event timing lives in streams.c (PHP API),
 * profiler start/stop lives in cuda.c, and NVTX range markers are
 * header-only (see profiler.cuh). This translation unit exists so the
 * build has a stable object file to link.
 */

extern "C" int php_cuda_profiler_unit_present(void) {
    return 1;
}
