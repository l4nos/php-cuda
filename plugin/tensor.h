#ifndef PHP_CUDA_TENSOR_H
#define PHP_CUDA_TENSOR_H

#include "php.h"
#include "php_cuda.h"
#include "tensor_kernels.cuh"

/*
 * CudaTensor: device-resident n-dimensional array.
 *
 * Memory model (mirrors ATen/c10): a refcounted Storage owns the device
 * allocation; a tensor object is a *view* onto that storage (offset, shape,
 * strides). reshape/transpose/slice create new views sharing the storage,
 * no device memory is copied.
 */

typedef struct _ct_storage {
    void *data;          /* device pointer */
    size_t nbytes;
    int device_id;
    uint32_t refcount;
} ct_storage;

typedef struct _ct_obj {
    ct_storage *storage;
    int64_t offset;                      /* element offset into storage */
    int ndim;
    int64_t shape[CT_MAX_DIMS];
    int64_t strides[CT_MAX_DIMS];        /* in elements */
    int dtype;                           /* CT_* */
    zend_object std;
} ct_obj;

static inline ct_obj *ct_obj_from_zobj(zend_object *obj) {
    return (ct_obj *)((char *)obj - XtOffsetOf(ct_obj, std));
}

static inline ct_obj *ct_obj_from_zval(zval *zv) {
    return ct_obj_from_zobj(Z_OBJ_P(zv));
}

/* Shared helpers (exported for nvrtc.c) */
ct_storage *ct_storage_new(size_t nbytes, int device_id);
void ct_storage_ref(ct_storage *s);
void ct_storage_unref(ct_storage *s);
int64_t ct_numel(int ndim, const int64_t *shape);
zend_bool ct_is_contiguous(ct_obj *t);
void ct_throw(const char *msg);
void ct_throw_cuda(cudaError_t err, const char *what);

#endif /* PHP_CUDA_TENSOR_H */
