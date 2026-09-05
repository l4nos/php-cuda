# Makefile fragment for the cuda extension.
#
# The PHP build system only knows how to compile C sources. CUDA (.cu) files
# are compiled here with NVCC into plain object files, which are then:
#   1. added as prerequisites of the module target (so they build first), and
#   2. appended to the link line via EXTRA_LDFLAGS (set in config.m4).

NVCC_OBJECTS = \
	cuda_kernels.o \
	tensor_kernels.o \
	memory_pool.o \
	memory_utils.o \
	cpu_ops.o \
	tensor_core_ops.o \
	profiler.o

# conv_ops.o is appended by config.m4 (via EXTRA_LDFLAGS) only when cuDNN is
# enabled; the pattern rule below compiles it the same way.

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -I$(top_srcdir) -c $< -o $@

# Additional prerequisites for the module target (recipe already exists in
# the generated Makefile; this only adds dependencies). CUDNN_EXTRA_OBJECTS
# is "conv_ops.o" when cuDNN is enabled, empty otherwise.
$(builddir)/cuda.la: $(NVCC_OBJECTS) $(CUDNN_EXTRA_OBJECTS)

clean-cuda:
	rm -f $(NVCC_OBJECTS) conv_ops.o

.PHONY: clean-cuda
