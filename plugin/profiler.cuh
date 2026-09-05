#ifndef PROFILER_CUH
#define PROFILER_CUH

#include <cuda_runtime.h>
#include "cuda_utils.cuh"

#if defined(HAVE_NVTX)
#  if defined(HAVE_NVTX3)
#    include <nvtx3/nvToolsExt.h>
#  else
#    include <nvToolsExt.h>
#  endif
#endif

/*
 * Profiling helpers. NVTX range markers compile to no-ops when the extension
 * is built without NVTX, so call sites never need their own guards.
 */

#ifdef HAVE_NVTX

class ProfilerMarker {
public:
    explicit ProfilerMarker(const char *name) { nvtxRangePushA(name); }
    ~ProfilerMarker() { nvtxRangePop(); }
};

#define PROFILE_SCOPE(name) ProfilerMarker __profiler_marker__(name)
#define PROFILE_FUNCTION() PROFILE_SCOPE(__FUNCTION__)

#else

#define PROFILE_SCOPE(name) ((void)0)
#define PROFILE_FUNCTION() ((void)0)

#endif /* HAVE_NVTX */

#endif // PROFILER_CUH
