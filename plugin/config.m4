dnl config.m4 for extension cuda
dnl
dnl Build options:
dnl   --with-cuda=DIR        CUDA toolkit root (default: autodetect from nvcc)
dnl   --with-cuda-arch=LIST  GPU architectures, e.g. "75;80;90" or "native" (default: native, fallback: sane list)
dnl   --with-cudnn=DIR       cuDNN root (default: same as CUDA; "no" disables)
dnl   --with-nvtx            Enable NVTX ranges (header-only NVTX3 preferred; optional)
dnl   --with-nvrtc=DIR       NVRTC + driver API for runtime kernel compilation (default: same as CUDA)
dnl   --enable-openmp        OpenMP for CPU fallback paths

PHP_ARG_WITH([cuda],
  [for CUDA support],
  [AS_HELP_STRING([--with-cuda=DIR],
    [Include CUDA support. DIR is the CUDA installation directory])],
  [no])

PHP_ARG_WITH([cuda-arch],
  [CUDA GPU architectures],
  [AS_HELP_STRING([--with-cuda-arch=LIST],
    [Semicolon-separated compute capabilities (e.g. "75;80;90") or "native"])],
  [native])

PHP_ARG_WITH([cudnn],
  [for cuDNN support],
  [AS_HELP_STRING([--with-cudnn=DIR],
    [Include cuDNN support. DIR is the cuDNN installation directory])],
  [yes])

PHP_ARG_WITH([nvtx],
  [for NVTX support],
  [AS_HELP_STRING([--with-nvtx],
    [Enable NVTX profiling ranges (optional, header-only NVTX3 supported)])],
  [no])

PHP_ARG_WITH([nvrtc],
  [for NVRTC support],
  [AS_HELP_STRING([--with-nvrtc=DIR],
    [Include NVRTC runtime kernel compilation. DIR defaults to the CUDA directory])],
  [yes])

PHP_ARG_ENABLE([openmp],
  [whether to enable OpenMP support],
  [AS_HELP_STRING([--enable-openmp],
    [Enable OpenMP support for CPU fallback])],
  [no])

if test "$PHP_CUDA" != "no"; then
    dnl -----------------------------------------------------------------
    dnl Locate the CUDA toolkit
    dnl -----------------------------------------------------------------
    if test "$PHP_CUDA" = "yes"; then
        AC_PATH_PROG(NVCC, nvcc, no)
        if test "$NVCC" = "no"; then
            AC_MSG_ERROR([Cannot find nvcc. Install the CUDA toolkit or pass --with-cuda=DIR])
        fi
        CUDA_DIR=`cd \`dirname "$NVCC"\`/.. && pwd`
    else
        CUDA_DIR=$PHP_CUDA
        AC_PATH_PROG(NVCC, nvcc, no, [$CUDA_DIR/bin:$PATH])
        if test "$NVCC" = "no"; then
            AC_MSG_ERROR([nvcc not found in $CUDA_DIR/bin])
        fi
    fi

    AC_MSG_CHECKING([for CUDA installation])
    if test ! -f "$CUDA_DIR/include/cuda_runtime.h"; then
        AC_MSG_ERROR([CUDA headers not found in $CUDA_DIR/include])
    fi
    AC_MSG_RESULT([$CUDA_DIR])

    dnl -----------------------------------------------------------------
    dnl Toolkit version (parsed from headers; no GPU required on build host)
    dnl -----------------------------------------------------------------
    AC_MSG_CHECKING([CUDA toolkit version])
    CUDART_VERSION=`awk '/^#define CUDART_VERSION/ {print $3}' "$CUDA_DIR/include/cuda_runtime_api.h" 2>/dev/null`
    if test -z "$CUDART_VERSION"; then
        CUDART_VERSION=`awk '/^#define CUDART_VERSION/ {print $3}' "$CUDA_DIR/include/cuda_runtime.h" 2>/dev/null`
    fi
    if test -z "$CUDART_VERSION"; then
        AC_MSG_ERROR([Could not determine CUDART_VERSION from headers])
    fi
    AC_MSG_RESULT([$CUDART_VERSION])
    if test "$CUDART_VERSION" -lt 11080; then
        AC_MSG_ERROR([CUDA 11.8 or higher is required (found $CUDART_VERSION)])
    fi
    AC_DEFINE_UNQUOTED([PHP_CUDA_TOOLKIT_VERSION], [$CUDART_VERSION], [CUDA toolkit version (CUDART_VERSION)])

    dnl -----------------------------------------------------------------
    dnl GPU architecture flags
    dnl -----------------------------------------------------------------
    AC_MSG_CHECKING([CUDA target architectures])
    if test "$PHP_CUDA_ARCH" = "yes"; then
        PHP_CUDA_ARCH="native"
    fi
    if test "$PHP_CUDA_ARCH" = "native"; then
        if test -x "$(command -v nvidia-smi)"; then
            ARCH_LIST=`nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d '.' | sort -u | tr '\n' ';' | sed 's/;$//'`
        fi
        if test -z "$ARCH_LIST"; then
            dnl No GPU visible on the build host: use a sane default for the toolkit
            if test "$CUDART_VERSION" -ge 12080; then
                ARCH_LIST="70;75;80;86;89;90;120"
            elif test "$CUDART_VERSION" -ge 12000; then
                ARCH_LIST="70;75;80;86;89;90"
            else
                ARCH_LIST="60;70;75;80;86;89;90"
            fi
            AC_MSG_NOTICE([no GPU detected; using default arch list: $ARCH_LIST])
        fi
    else
        ARCH_LIST="$PHP_CUDA_ARCH"
    fi

    GENCODE_FLAGS=""
    SAVE_IFS=$IFS; IFS=';'
    for arch in $ARCH_LIST; do
        case "$arch" in
            ''|*[!0-9]*)
                AC_MSG_ERROR([Invalid architecture "$arch" in --with-cuda-arch (expected e.g. "75;80;90")])
                ;;
        esac
        dnl CUDA 12 dropped everything below sm_50.
        if test "$arch" -lt 50; then
            AC_MSG_ERROR([sm_$arch is not supported by CUDA >= 12. Use sm_70 or newer.])
        fi
        GENCODE_FLAGS="$GENCODE_FLAGS -gencode arch=compute_$arch,code=sm_$arch"
    done
    IFS=$SAVE_IFS
    AC_MSG_RESULT([$ARCH_LIST])

    NVCC_FLAGS="-O3 -std=c++17 -Xcompiler -fPIC --expt-relaxed-constexpr $GENCODE_FLAGS"

    dnl -----------------------------------------------------------------
    dnl CUDA libraries
    dnl -----------------------------------------------------------------
    for libdir in "$CUDA_DIR/lib64" "$CUDA_DIR/lib" "$CUDA_DIR/lib/x86_64-linux-gnu"; do
        if test -d "$libdir"; then
            CUDA_LIBDIR="$libdir"
            break
        fi
    done
    if test -z "$CUDA_LIBDIR"; then
        AC_MSG_ERROR([CUDA library directory not found under $CUDA_DIR])
    fi

    PHP_ADD_INCLUDE($CUDA_DIR/include)
    PHP_ADD_LIBRARY_WITH_PATH(cudart, $CUDA_LIBDIR, CUDA_SHARED_LIBADD)
    PHP_ADD_LIBRARY_WITH_PATH(cublas, $CUDA_LIBDIR, CUDA_SHARED_LIBADD)

    dnl -----------------------------------------------------------------
    dnl cuDNN (optional but recommended)
    dnl -----------------------------------------------------------------
    CUDNN_EXTRA_SOURCES=""
    CUDNN_EXTRA_OBJECTS=""
    if test "$PHP_CUDNN" != "no"; then
        if test "$PHP_CUDNN" = "yes"; then
            CUDNN_DIR=$CUDA_DIR
        else
            CUDNN_DIR=$PHP_CUDNN
        fi

        AC_MSG_CHECKING([for cuDNN])
        CUDNN_FOUND=no
        for incdir in "$CUDNN_DIR/include" "$CUDA_DIR/include" /usr/include; do
            if test -f "$incdir/cudnn.h"; then
                CUDNN_INCLUDE="$incdir"
                CUDNN_FOUND=yes
                break
            fi
        done

        if test "$CUDNN_FOUND" = "yes"; then
            CUDNN_MAJOR=`awk '/^#define CUDNN_MAJOR/ {print $3}' "$CUDNN_INCLUDE/cudnn_version.h" 2>/dev/null`
            if test -z "$CUDNN_MAJOR"; then
                CUDNN_MAJOR=`awk '/^#define CUDNN_MAJOR/ {print $3}' "$CUDNN_INCLUDE/cudnn.h" 2>/dev/null`
            fi
            PHP_ADD_INCLUDE($CUDNN_INCLUDE)
            for cudnnlibdir in "$CUDNN_DIR/lib64" "$CUDNN_DIR/lib" "$CUDA_LIBDIR" /usr/lib/x86_64-linux-gnu; do
                if test -f "$cudnnlibdir/libcudnn.so"; then
                    PHP_ADD_LIBRARY_WITH_PATH(cudnn, $cudnnlibdir, CUDA_SHARED_LIBADD)
                    break
                fi
            done
            AC_DEFINE(HAVE_CUDNN, 1, [Whether you have cuDNN])
            CUDNN_EXTRA_SOURCES="cudnn_ops.c"
            CUDNN_EXTRA_OBJECTS="conv_ops.o"
            AC_MSG_RESULT([found, major version $CUDNN_MAJOR])
        else
            AC_MSG_RESULT([not found, building without cuDNN])
            AC_MSG_NOTICE([cuDNN not found; convolution/cuDNN bindings will be stubs])
        fi
    fi

    dnl -----------------------------------------------------------------
    dnl NVTX (optional; NVTX3 is header-only on CUDA >= 12)
    dnl -----------------------------------------------------------------
    if test "$PHP_NVTX" != "no"; then
        AC_MSG_CHECKING([for NVTX])
        if test -f "$CUDA_DIR/include/nvtx3/nvToolsExt.h"; then
            AC_DEFINE(HAVE_NVTX, 1, [Whether you have NVTX])
            AC_DEFINE(HAVE_NVTX3, 1, [NVTX3 header-only variant])
            AC_MSG_RESULT([found (NVTX3, header-only)])
        elif test -f "$CUDA_DIR/include/nvToolsExt.h"; then
            PHP_ADD_LIBRARY_WITH_PATH(nvToolsExt, $CUDA_LIBDIR, CUDA_SHARED_LIBADD)
            AC_DEFINE(HAVE_NVTX, 1, [Whether you have NVTX])
            AC_MSG_RESULT([found (legacy nvToolsExt)])
        else
            AC_MSG_RESULT([not found, profiling ranges disabled])
        fi
    fi

    dnl -----------------------------------------------------------------
    dnl NVRTC + driver API (runtime kernel compilation)
    dnl -----------------------------------------------------------------
    NVRTC_EXTRA_SOURCES=""
    if test "$PHP_NVRTC" != "no"; then
        if test "$PHP_NVRTC" = "yes"; then
            NVRTC_DIR=$CUDA_DIR
        else
            NVRTC_DIR=$PHP_NVRTC
        fi

        AC_MSG_CHECKING([for NVRTC])
        if test -f "$NVRTC_DIR/include/nvrtc.h"; then
            PHP_ADD_INCLUDE($NVRTC_DIR/include)
            PHP_ADD_LIBRARY_WITH_PATH(nvrtc, $CUDA_LIBDIR, CUDA_SHARED_LIBADD)
            dnl The driver API (libcuda) is deliberately NOT linked: it belongs
            dnl to the driver, not the toolkit, and is absent on GPU-less build
            dnl hosts. nvrtc.c resolves it lazily with dlopen at runtime.
            PHP_ADD_LIBRARY(dl, 1, CUDA_SHARED_LIBADD)
            AC_DEFINE(HAVE_NVRTC, 1, [Whether you have NVRTC])
            NVRTC_EXTRA_SOURCES="nvrtc.c"
            AC_MSG_RESULT([found])
        else
            AC_MSG_RESULT([not found, runtime kernel compilation disabled])
        fi
    fi

    dnl -----------------------------------------------------------------
    dnl OpenMP (optional, CPU fallback)
    dnl -----------------------------------------------------------------
    if test "$PHP_OPENMP" != "no"; then
        AC_MSG_CHECKING([for OpenMP support])
        AC_LANG_PUSH([C])
        ORIG_CFLAGS="$CFLAGS"
        CFLAGS="$CFLAGS -fopenmp"
        AC_TRY_LINK(
            [#include <omp.h>],
            [omp_get_num_threads();],
            [
                AC_MSG_RESULT([yes])
                AC_DEFINE(HAVE_OPENMP, 1, [Whether you have OpenMP])
                EXTRA_CFLAGS="$EXTRA_CFLAGS -fopenmp"
                EXTRA_LDFLAGS="$EXTRA_LDFLAGS -fopenmp"
                NVCC_FLAGS="$NVCC_FLAGS -Xcompiler -fopenmp"
            ],
            [
                AC_MSG_RESULT([no])
            ]
        )
        CFLAGS="$ORIG_CFLAGS"
        AC_LANG_POP([C])
    fi

    dnl -----------------------------------------------------------------
    dnl Sources
    dnl -----------------------------------------------------------------
    dnl .cu files are compiled by NVCC via Makefile.frag into plain .o files.
    dnl They are appended to the link through EXTRA_LDFLAGS and made a
    dnl prerequisite of the module target in Makefile.frag.
    NVCC_OBJECTS="cuda_kernels.o tensor_kernels.o memory_pool.o memory_utils.o cpu_ops.o tensor_core_ops.o profiler.o $CUDNN_EXTRA_OBJECTS"

    EXTRA_LDFLAGS="$EXTRA_LDFLAGS $NVCC_OBJECTS"

    PHP_SUBST(NVCC)
    PHP_SUBST(NVCC_FLAGS)
    PHP_SUBST(CUDA_DIR)
    PHP_SUBST(CUDA_SHARED_LIBADD)
    PHP_SUBST(CUDNN_EXTRA_OBJECTS)

    dnl PHP_NEW_EXTENSION must come first: it sets ext_srcdir/ext_builddir,
    dnl which PHP_ADD_MAKEFILE_FRAGMENT needs to locate Makefile.frag.
    PHP_NEW_EXTENSION(cuda, cuda.c tensor.c streams.c cublas_ops.c $NVRTC_EXTRA_SOURCES $CUDNN_EXTRA_SOURCES, $ext_shared)

    PHP_ADD_MAKEFILE_FRAGMENT
fi
