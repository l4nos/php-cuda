#include "php.h"
#include "zend_exceptions.h"
#include "zend_smart_str.h"
#include "tensor.h"
#include "cuda_kernels.cuh"

zend_class_entry *cuda_tensor_ce;
static zend_object_handlers ct_handlers;

/* -------------------------------------------------------------------------
 * Storage
 * ------------------------------------------------------------------------- */
ct_storage *ct_storage_new(size_t nbytes, int device_id) {
    ct_storage *s = emalloc(sizeof(ct_storage));
    s->data = NULL;
    s->nbytes = nbytes;
    s->device_id = device_id;
    s->refcount = 1;

    if (nbytes > 0) {
        if (cuda_use_device(device_id) != cudaSuccess) {
            efree(s);
            return NULL;
        }
        if (cudaMalloc(&s->data, nbytes) != cudaSuccess) {
            efree(s);
            return NULL;
        }
    }
    return s;
}

void ct_storage_ref(ct_storage *s) {
    if (s) s->refcount++;
}

void ct_storage_unref(ct_storage *s) {
    if (!s) return;
    if (--s->refcount == 0) {
        if (s->data) {
            cudaSetDevice(s->device_id);
            cudaFree(s->data);
        }
        efree(s);
    }
}

/* -------------------------------------------------------------------------
 * Small utilities
 * ------------------------------------------------------------------------- */
int64_t ct_numel(int ndim, const int64_t *shape) {
    int64_t n = 1;
    for (int i = 0; i < ndim; i++) {
        if (shape[i] < 0) return -1;
        n *= shape[i];
    }
    return n;
}

zend_bool ct_is_contiguous(ct_obj *t) {
    int64_t expected = 1;
    for (int d = t->ndim - 1; d >= 0; d--) {
        if (t->shape[d] == 1) continue; /* stride is irrelevant for size-1 dims */
        if (t->strides[d] != expected) return 0;
        expected *= t->shape[d];
    }
    return 1;
}

void ct_throw(const char *msg) {
    zend_throw_exception(cuda_exception_ce, msg, 0);
}

void ct_throw_cuda(cudaError_t err, const char *what) {
    zend_throw_exception(cuda_exception_ce, (char *)what, (zend_long)err);
}

static const char *ct_dtype_name(int dtype) {
    switch (dtype) {
        case CT_FP32:  return "fp32";
        case CT_FP64:  return "fp64";
        case CT_INT32: return "int32";
        case CT_FP16:  return "fp16";
        case CT_BF16:  return "bf16";
        case CT_INT8:  return "int8";
        default:       return "unknown";
    }
}

static int ct_dtype_valid(int dtype) {
    return ct_dtype_size(dtype) > 0;
}

/* Contiguous strides for a shape. */
static void ct_default_strides(int ndim, const int64_t *shape, int64_t *strides) {
    int64_t acc = 1;
    for (int d = ndim - 1; d >= 0; d--) {
        strides[d] = acc;
        acc *= shape[d];
    }
}

/* Element pointer for a view. */
static inline void *ct_data(ct_obj *t) {
    return (char *)t->storage->data + (size_t)t->offset * ct_dtype_size(t->dtype);
}

/* -------------------------------------------------------------------------
 * Object lifecycle
 * ------------------------------------------------------------------------- */
static zend_object *ct_create_object(zend_class_entry *ce) {
    ct_obj *t = ecalloc(1, sizeof(ct_obj) + zend_object_properties_size(ce));
    zend_object_std_init(&t->std, ce);
    object_properties_init(&t->std, ce);
    t->std.handlers = &ct_handlers;
    t->storage = NULL;
    return &t->std;
}

static void ct_free_object(zend_object *obj) {
    ct_obj *t = ct_obj_from_zobj(obj);
    ct_storage_unref(t->storage);
    zend_object_std_dtor(&t->std);
}

static zend_object *ct_clone_object(zend_object *old_obj) {
    ct_obj *old_t = ct_obj_from_zobj(old_obj);
    zend_object *new_obj = ct_create_object(old_t->std.ce);
    ct_obj *new_t = ct_obj_from_zobj(new_obj);

    new_t->storage = old_t->storage;
    ct_storage_ref(new_t->storage);
    new_t->offset = old_t->offset;
    new_t->ndim = old_t->ndim;
    new_t->dtype = old_t->dtype;
    memcpy(new_t->shape, old_t->shape, sizeof(new_t->shape));
    memcpy(new_t->strides, old_t->strides, sizeof(new_t->strides));

    zend_objects_clone_members(&new_t->std, &old_t->std);
    return new_obj;
}

/* Create a new tensor object owning a fresh contiguous allocation. */
static ct_obj *ct_alloc_tensor(int ndim, const int64_t *shape, int dtype, int device_id) {
    zval zv;
    object_init_ex(&zv, cuda_tensor_ce);
    ct_obj *t = ct_obj_from_zval(&zv);

    int64_t numel = ct_numel(ndim, shape);
    size_t nbytes = (size_t)numel * ct_dtype_size(dtype);

    t->storage = ct_storage_new(nbytes, device_id);
    if (!t->storage) {
        zval_ptr_dtor(&zv);
        ct_throw("CudaTensor: device allocation failed");
        return NULL;
    }
    t->offset = 0;
    t->ndim = ndim;
    t->dtype = dtype;
    memset(t->shape, 0, sizeof(t->shape));
    memset(t->strides, 0, sizeof(t->strides));
    memcpy(t->shape, shape, sizeof(int64_t) * ndim);
    ct_default_strides(ndim, shape, t->strides);
    return t;
}

/* Create a view sharing another tensor's storage. */
static ct_obj *ct_view_of(ct_obj *base) {
    zval zv;
    object_init_ex(&zv, cuda_tensor_ce);
    ct_obj *t = ct_obj_from_zval(&zv);

    t->storage = base->storage;
    ct_storage_ref(t->storage);
    t->offset = base->offset;
    t->ndim = base->ndim;
    t->dtype = base->dtype;
    memcpy(t->shape, base->shape, sizeof(t->shape));
    memcpy(t->strides, base->strides, sizeof(t->strides));
    return t;
}

/* Wrap a ct_obj* (created via object_init_ex internally) into return_value.
 * ct_alloc_tensor/ct_view_of create temp zvals we never see; these helpers
 * instead return the object and we re-wrap. To keep ownership simple the
 * helpers above intentionally leak the temp zval's only reference into the
 * caller, so here we reconstruct a zval pointing at the same object. */
static void ct_return_obj(zval *return_value, ct_obj *t) {
    ZVAL_OBJ(return_value, &t->std);
    /* The temp zval from object_init_ex held one ref; the caller's helpers
     * abandoned it, so the object currently has refcount 1 which we adopt. */
}

/* Drop the single creation reference of an object returned by
 * ct_alloc_tensor/ct_view_of/ct_make_contiguous (error paths and after-use
 * cleanup). Runs free_obj, which releases the storage reference. */
static void ct_release(ct_obj *t) {
    zval zv;
    ZVAL_OBJ(&zv, &t->std);
    zval_ptr_dtor(&zv);
}

/* -------------------------------------------------------------------------
 * Argument helpers
 * ------------------------------------------------------------------------- */
static int ct_parse_shape(zval *arr, int64_t *shape, int *ndim) {
    HashTable *ht = Z_ARRVAL_P(arr);
    int n = zend_hash_num_elements(ht);
    if (n < 1 || n > CT_MAX_DIMS) return FAILURE;

    int i = 0;
    zval *zv;
    ZEND_HASH_FOREACH_VAL(ht, zv) {
        zend_long v = zval_get_long(zv);
        if (v < 0) return FAILURE;
        shape[i++] = (int64_t)v;
    } ZEND_HASH_FOREACH_END();
    *ndim = n;
    return SUCCESS;
}

/* -------------------------------------------------------------------------
 * Broadcasting
 * ------------------------------------------------------------------------- */
static int ct_broadcast(ct_obj *a, ct_obj *b,
                        int64_t *out_shape, int *out_ndim,
                        ct_dims *sa, ct_dims *sb) {
    int ndim = a->ndim > b->ndim ? a->ndim : b->ndim;

    for (int i = 0; i < ndim; i++) {
        int ai = a->ndim - 1 - i;
        int bi = b->ndim - 1 - i;
        int oi = ndim - 1 - i;

        int64_t ad = ai >= 0 ? a->shape[ai] : 1;
        int64_t bd = bi >= 0 ? b->shape[bi] : 1;
        int64_t as = ai >= 0 ? a->strides[ai] : 0;
        int64_t bs = bi >= 0 ? b->strides[bi] : 0;

        if (ad == bd) {
            out_shape[oi] = ad;
            sa->v[oi] = as;
            sb->v[oi] = bs;
        } else if (ad == 1) {
            out_shape[oi] = bd;
            sa->v[oi] = 0;
            sb->v[oi] = bs;
        } else if (bd == 1) {
            out_shape[oi] = ad;
            sa->v[oi] = as;
            sb->v[oi] = 0;
        } else {
            return FAILURE;
        }
    }
    *out_ndim = ndim;
    return SUCCESS;
}

/* Wrap a PHP scalar as a 0-stride broadcast tensor view of a 1-element
 * device buffer. The returned storage must be freed by the caller. */
typedef struct _ct_scalar {
    ct_obj view;
    ct_storage storage;
    double host_value;
    void *dev_value;
} ct_scalar;

static int ct_scalar_init(ct_scalar *s, double value, int dtype, int device_id) {
    memset(s, 0, sizeof(*s));
    s->dev_value = NULL;

    if (cuda_use_device(device_id) != cudaSuccess) return FAILURE;
    if (cudaMalloc(&s->dev_value, ct_dtype_size(dtype)) != cudaSuccess) return FAILURE;

    /* Stage as fp32, cast on device to the target dtype. */
    float f = (float)value;
    void *staging = NULL;
    if (cudaMalloc(&staging, sizeof(float)) != cudaSuccess) {
        cudaFree(s->dev_value);
        return FAILURE;
    }
    cudaMemcpy(staging, &f, sizeof(float), cudaMemcpyHostToDevice);
    cudaError_t err = ct_copy_cast(staging, s->dev_value, 1, CT_FP32, dtype, 0);
    cudaFree(staging);
    if (err != cudaSuccess) {
        cudaFree(s->dev_value);
        return FAILURE;
    }

    s->storage.data = s->dev_value;
    s->storage.nbytes = ct_dtype_size(dtype);
    s->storage.device_id = device_id;
    s->storage.refcount = 1;

    s->view.storage = &s->storage;
    s->view.offset = 0;
    s->view.ndim = 1;
    s->view.dtype = dtype;
    memset(s->view.shape, 0, sizeof(s->view.shape));
    memset(s->view.strides, 0, sizeof(s->view.strides));
    s->view.shape[0] = 1;
    s->view.strides[0] = 0;
    return SUCCESS;
}

static void ct_scalar_destroy(ct_scalar *s) {
    if (s->dev_value) cudaFree(s->dev_value);
}

/* -------------------------------------------------------------------------
 * Gather helper: produce a contiguous tensor from any view.
 * ------------------------------------------------------------------------- */
static ct_obj *ct_make_contiguous(ct_obj *src) {
    if (ct_is_contiguous(src)) {
        ct_obj *t = ct_view_of(src);
        return t;
    }
    ct_obj *out = ct_alloc_tensor(src->ndim, src->shape, src->dtype, src->storage->device_id);
    if (!out) return NULL;

    ct_dims shape, strides;
    memset(&shape, 0, sizeof(shape));
    memset(&strides, 0, sizeof(strides));
    memcpy(shape.v, src->shape, sizeof(int64_t) * src->ndim);
    memcpy(strides.v, src->strides, sizeof(int64_t) * src->ndim);

    int64_t total = ct_numel(src->ndim, src->shape);
    cudaError_t err = ct_gather(ct_data(src), strides, ct_data(out),
                                shape, src->ndim, total, src->dtype, 0);
    if (err != cudaSuccess) {
        ct_release(out);
        ct_throw_cuda(err, "CudaTensor: gather failed");
        return NULL;
    }
    cudaDeviceSynchronize();
    return out;
}

/* -------------------------------------------------------------------------
 * Host <-> device
 * ------------------------------------------------------------------------- */

/* Recursively infer shape and flatten a PHP array into doubles. */
static int ct_flatten_array(zval *arr, int depth, int64_t *shape, int *ndim,
                            double **buf, size_t *len, size_t *cap) {
    if (depth >= CT_MAX_DIMS) return FAILURE;
    HashTable *ht = Z_ARRVAL_P(arr);
    zend_long n = zend_hash_num_elements(ht);

    if (depth >= *ndim) {
        shape[depth] = n;
        *ndim = depth + 1;
    } else if (shape[depth] != n) {
        return FAILURE; /* ragged */
    }

    zval *zv;
    zend_bool first = 1;
    zend_bool has_children = 0;
    ZEND_HASH_FOREACH_VAL(ht, zv) {
        if (first) {
            has_children = (Z_TYPE_P(zv) == IS_ARRAY);
            first = 0;
        }
        if (has_children != (Z_TYPE_P(zv) == IS_ARRAY)) return FAILURE;

        if (Z_TYPE_P(zv) == IS_ARRAY) {
            if (ct_flatten_array(zv, depth + 1, shape, ndim, buf, len, cap) == FAILURE) {
                return FAILURE;
            }
        } else {
            if (*len >= *cap) {
                *cap = *cap ? *cap * 2 : 256;
                *buf = erealloc(*buf, *cap * sizeof(double));
            }
            (*buf)[(*len)++] = zval_get_double(zv);
        }
    } ZEND_HASH_FOREACH_END();
    return SUCCESS;
}

static void ct_build_nested(const float *data, int ndim, const int64_t *shape,
                            int depth, size_t *pos, zval *out) {
    array_init_size(out, (uint32_t)shape[depth]);
    for (int64_t i = 0; i < shape[depth]; i++) {
        if (depth == ndim - 1) {
            add_next_index_double(out, (double)data[(*pos)++]);
        } else {
            zval child;
            ct_build_nested(data, ndim, shape, depth + 1, pos, &child);
            add_next_index_zval(out, &child);
        }
    }
}

/* Copy tensor contents into a host fp32 buffer (emalloc'ed). */
static float *ct_to_host_f32(ct_obj *t, int64_t *total_out) {
    ct_obj *contig = ct_make_contiguous(t);
    if (!contig) return NULL;

    int64_t total = ct_numel(contig->ndim, contig->shape);
    float *host = emalloc((size_t)total * sizeof(float));

    cudaError_t err = cudaSuccess;
    if (contig->dtype == CT_FP32) {
        err = cudaMemcpy(host, ct_data(contig), (size_t)total * sizeof(float), cudaMemcpyDeviceToHost);
    } else {
        void *staging = NULL;
        err = cudaMalloc(&staging, (size_t)total * sizeof(float));
        if (err == cudaSuccess) {
            err = ct_copy_cast(ct_data(contig), staging, total, contig->dtype, CT_FP32, 0);
            if (err == cudaSuccess) {
                err = cudaMemcpy(host, staging, (size_t)total * sizeof(float), cudaMemcpyDeviceToHost);
            }
            cudaFree(staging);
        }
    }

    ct_release(contig);
    if (err != cudaSuccess) {
        efree(host);
        ct_throw_cuda(err, "CudaTensor: device-to-host copy failed");
        return NULL;
    }
    *total_out = total;
    return host;
}

/* -------------------------------------------------------------------------
 * Elementwise op plumbing
 * ------------------------------------------------------------------------- */
static void ct_binary_op(INTERNAL_FUNCTION_PARAMETERS, int op) {
    zval *arg;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ZVAL(arg)
    ZEND_PARSE_PARAMETERS_END();

    ct_obj *a = ct_obj_from_zval(getThis());
    ct_obj *b = NULL;
    ct_scalar scalar;
    int using_scalar = 0;

    if (Z_TYPE_P(arg) == IS_OBJECT && Z_OBJCE_P(arg) == cuda_tensor_ce) {
        b = ct_obj_from_zval(arg);
        if (b->dtype != a->dtype) {
            ct_throw("CudaTensor: dtype mismatch in binary op");
            RETURN_THROWS();
        }
        if (b->storage->device_id != a->storage->device_id) {
            ct_throw("CudaTensor: operands are on different devices (move one with ->toDevice() when multi-GPU support lands)");
            RETURN_THROWS();
        }
    } else if (Z_TYPE_P(arg) == IS_LONG || Z_TYPE_P(arg) == IS_DOUBLE) {
        if (ct_scalar_init(&scalar, zval_get_double(arg), a->dtype, a->storage->device_id) == FAILURE) {
            ct_throw("CudaTensor: failed to stage scalar operand");
            RETURN_THROWS();
        }
        b = &scalar.view;
        using_scalar = 1;
    } else {
        ct_throw("CudaTensor: operand must be a CudaTensor or a number");
        RETURN_THROWS();
    }

    int64_t out_shape[CT_MAX_DIMS];
    int out_ndim;
    ct_dims sa, sb;
    memset(&sa, 0, sizeof(sa));
    memset(&sb, 0, sizeof(sb));

    if (ct_broadcast(a, b, out_shape, &out_ndim, &sa, &sb) == FAILURE) {
        if (using_scalar) ct_scalar_destroy(&scalar);
        ct_throw("CudaTensor: shapes are not broadcast-compatible");
        RETURN_THROWS();
    }

    ct_obj *out = ct_alloc_tensor(out_ndim, out_shape, a->dtype, a->storage->device_id);
    if (!out) {
        if (using_scalar) ct_scalar_destroy(&scalar);
        RETURN_THROWS();
    }

    ct_dims shape;
    memset(&shape, 0, sizeof(shape));
    memcpy(shape.v, out_shape, sizeof(int64_t) * out_ndim);
    int64_t total = ct_numel(out_ndim, out_shape);

    cudaError_t err = ct_elementwise_binary(op, ct_data(a), sa, ct_data(b), sb,
                                            ct_data(out), shape, out_ndim, total,
                                            a->dtype, 0);
    if (using_scalar) ct_scalar_destroy(&scalar);

    if (err != cudaSuccess) {
        ct_release(out);
        ct_throw_cuda(err, "CudaTensor: elementwise kernel failed");
        RETURN_THROWS();
    }
    cudaDeviceSynchronize();
    ct_return_obj(return_value, out);
}

static void ct_binary_op_inplace(INTERNAL_FUNCTION_PARAMETERS, int op) {
    zval *arg;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ZVAL(arg)
    ZEND_PARSE_PARAMETERS_END();

    ct_obj *a = ct_obj_from_zval(getThis());
    if (!ct_is_contiguous(a)) {
        ct_throw("CudaTensor: in-place ops require a contiguous tensor");
        RETURN_THROWS();
    }

    ct_obj *b = NULL;
    ct_scalar scalar;
    int using_scalar = 0;

    if (Z_TYPE_P(arg) == IS_OBJECT && Z_OBJCE_P(arg) == cuda_tensor_ce) {
        b = ct_obj_from_zval(arg);
        if (b->dtype != a->dtype || b->storage->device_id != a->storage->device_id) {
            ct_throw("CudaTensor: dtype or device mismatch in in-place op");
            RETURN_THROWS();
        }
    } else if (Z_TYPE_P(arg) == IS_LONG || Z_TYPE_P(arg) == IS_DOUBLE) {
        if (ct_scalar_init(&scalar, zval_get_double(arg), a->dtype, a->storage->device_id) == FAILURE) {
            ct_throw("CudaTensor: failed to stage scalar operand");
            RETURN_THROWS();
        }
        b = &scalar.view;
        using_scalar = 1;
    } else {
        ct_throw("CudaTensor: operand must be a CudaTensor or a number");
        RETURN_THROWS();
    }

    int64_t out_shape[CT_MAX_DIMS];
    int out_ndim;
    ct_dims sa, sb;
    memset(&sa, 0, sizeof(sa));
    memset(&sb, 0, sizeof(sb));

    if (ct_broadcast(a, b, out_shape, &out_ndim, &sa, &sb) == FAILURE ||
        out_ndim != a->ndim || memcmp(out_shape, a->shape, sizeof(int64_t) * a->ndim) != 0) {
        if (using_scalar) ct_scalar_destroy(&scalar);
        ct_throw("CudaTensor: operand cannot broadcast to this tensor's shape in-place");
        RETURN_THROWS();
    }

    ct_dims shape;
    memset(&shape, 0, sizeof(shape));
    memcpy(shape.v, a->shape, sizeof(int64_t) * a->ndim);
    int64_t total = ct_numel(a->ndim, a->shape);

    cudaError_t err = ct_elementwise_binary(op, ct_data(a), sa, ct_data(b), sb,
                                            ct_data(a), shape, a->ndim, total,
                                            a->dtype, 0);
    if (using_scalar) ct_scalar_destroy(&scalar);

    if (err != cudaSuccess) {
        ct_throw_cuda(err, "CudaTensor: in-place kernel failed");
        RETURN_THROWS();
    }
    cudaDeviceSynchronize();
    RETURN_ZVAL(getThis(), 1, 0);
}

static void ct_unary_op(INTERNAL_FUNCTION_PARAMETERS, int op) {
    ZEND_PARSE_PARAMETERS_NONE();

    ct_obj *a = ct_obj_from_zval(getThis());
    ct_obj *out = ct_alloc_tensor(a->ndim, a->shape, a->dtype, a->storage->device_id);
    if (!out) RETURN_THROWS();

    ct_dims shape, strides;
    memset(&shape, 0, sizeof(shape));
    memset(&strides, 0, sizeof(strides));
    memcpy(shape.v, a->shape, sizeof(int64_t) * a->ndim);
    memcpy(strides.v, a->strides, sizeof(int64_t) * a->ndim);

    int64_t total = ct_numel(a->ndim, a->shape);
    cudaError_t err = ct_elementwise_unary(op, ct_data(a), strides, ct_data(out),
                                           shape, a->ndim, total, a->dtype, 0);
    if (err != cudaSuccess) {
        ct_release(out);
        ct_throw_cuda(err, "CudaTensor: unary kernel failed");
        RETURN_THROWS();
    }
    cudaDeviceSynchronize();
    ct_return_obj(return_value, out);
}

static void ct_reduce_op(INTERNAL_FUNCTION_PARAMETERS, int op) {
    ZEND_PARSE_PARAMETERS_NONE();

    ct_obj *a = ct_obj_from_zval(getThis());

    ct_dims shape, strides;
    memset(&shape, 0, sizeof(shape));
    memset(&strides, 0, sizeof(strides));
    memcpy(shape.v, a->shape, sizeof(int64_t) * a->ndim);
    memcpy(strides.v, a->strides, sizeof(int64_t) * a->ndim);

    int64_t total = ct_numel(a->ndim, a->shape);
    double result = 0.0;
    cudaError_t err = ct_reduce(op, ct_data(a), strides, shape, a->ndim, total,
                                a->dtype, &result, 0);
    if (err != cudaSuccess) {
        ct_throw_cuda(err, "CudaTensor: reduction failed");
        RETURN_THROWS();
    }
    RETURN_DOUBLE(result);
}

/* -------------------------------------------------------------------------
 * Static constructors
 * ------------------------------------------------------------------------- */
ZEND_METHOD(CudaTensor, fromArray) {
    zval *arr;
    zend_long dtype = CT_FP32;
    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_ARRAY(arr)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(dtype)
    ZEND_PARSE_PARAMETERS_END();

    if (!ct_dtype_valid((int)dtype)) {
        ct_throw("CudaTensor: invalid dtype");
        RETURN_THROWS();
    }

    int64_t shape[CT_MAX_DIMS] = {0};
    int ndim = 0;
    double *buf = NULL;
    size_t len = 0, cap = 0;

    if (ct_flatten_array(arr, 0, shape, &ndim, &buf, &len, &cap) == FAILURE || ndim == 0) {
        if (buf) efree(buf);
        ct_throw("CudaTensor::fromArray: array must be rectangular and non-empty");
        RETURN_THROWS();
    }

    int64_t total = ct_numel(ndim, shape);
    if (total != (int64_t)len) {
        efree(buf);
        ct_throw("CudaTensor::fromArray: ragged array");
        RETURN_THROWS();
    }

    ct_obj *t = ct_alloc_tensor(ndim, shape, (int)dtype, CUDA_G(current_device));
    if (!t) {
        efree(buf);
        RETURN_THROWS();
    }

    /* Stage as fp32 on device, cast if needed. */
    float *fbuf = emalloc((size_t)total * sizeof(float));
    for (size_t i = 0; i < len; i++) fbuf[i] = (float)buf[i];
    efree(buf);

    cudaError_t err = cudaSuccess;
    if (dtype == CT_FP32) {
        err = cudaMemcpy(ct_data(t), fbuf, (size_t)total * sizeof(float), cudaMemcpyHostToDevice);
    } else {
        void *staging = NULL;
        err = cudaMalloc(&staging, (size_t)total * sizeof(float));
        if (err == cudaSuccess) {
            err = cudaMemcpy(staging, fbuf, (size_t)total * sizeof(float), cudaMemcpyHostToDevice);
            if (err == cudaSuccess) {
                err = ct_copy_cast(staging, ct_data(t), total, CT_FP32, (int)dtype, 0);
            }
            cudaFree(staging);
        }
    }
    efree(fbuf);

    if (err != cudaSuccess) {
        ct_release(t);
        ct_throw_cuda(err, "CudaTensor::fromArray: host-to-device copy failed");
        RETURN_THROWS();
    }
    ct_return_obj(return_value, t);
}

static void ct_fill_constructor(INTERNAL_FUNCTION_PARAMETERS, double value, int has_value) {
    zval *shape_zv;
    zend_long dtype = CT_FP32;
    ZEND_PARSE_PARAMETERS_START(has_value ? 2 : 1, has_value ? 3 : 2)
        Z_PARAM_ARRAY(shape_zv)
        if (has_value) { Z_PARAM_DOUBLE(value) }
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(dtype)
    ZEND_PARSE_PARAMETERS_END();

    if (!ct_dtype_valid((int)dtype)) {
        ct_throw("CudaTensor: invalid dtype");
        RETURN_THROWS();
    }

    int64_t shape[CT_MAX_DIMS];
    int ndim;
    if (ct_parse_shape(shape_zv, shape, &ndim) == FAILURE) {
        ct_throw("CudaTensor: invalid shape (1-8 non-negative dimensions)");
        RETURN_THROWS();
    }

    ct_obj *t = ct_alloc_tensor(ndim, shape, (int)dtype, CUDA_G(current_device));
    if (!t) RETURN_THROWS();

    int64_t total = ct_numel(ndim, shape);
    cudaError_t err = ct_fill(ct_data(t), value, total, (int)dtype, 0);
    if (err != cudaSuccess) {
        ct_release(t);
        ct_throw_cuda(err, "CudaTensor: fill failed");
        RETURN_THROWS();
    }
    cudaDeviceSynchronize();
    ct_return_obj(return_value, t);
}

ZEND_METHOD(CudaTensor, zeros) {
    ct_fill_constructor(INTERNAL_FUNCTION_PARAM_PASSTHRU, 0.0, 0);
}

ZEND_METHOD(CudaTensor, ones) {
    ct_fill_constructor(INTERNAL_FUNCTION_PARAM_PASSTHRU, 1.0, 0);
}

ZEND_METHOD(CudaTensor, full) {
    double value = 0.0;
    ct_fill_constructor(INTERNAL_FUNCTION_PARAM_PASSTHRU, value, 1);
}

ZEND_METHOD(CudaTensor, rand) {
    zval *shape_zv;
    zend_long dtype = CT_FP32;
    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_ARRAY(shape_zv)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(dtype)
    ZEND_PARSE_PARAMETERS_END();

    if (!ct_dtype_valid((int)dtype)) {
        ct_throw("CudaTensor: invalid dtype");
        RETURN_THROWS();
    }

    int64_t shape[CT_MAX_DIMS];
    int ndim;
    if (ct_parse_shape(shape_zv, shape, &ndim) == FAILURE) {
        ct_throw("CudaTensor: invalid shape");
        RETURN_THROWS();
    }

    int64_t total = ct_numel(ndim, shape);
    float *host = emalloc((size_t)total * sizeof(float));
    for (int64_t i = 0; i < total; i++) {
        /* Uniform [0,1). Simple host-side RNG; a cuRAND-backed generator is
         * on the roadmap (Phase 5). */
        host[i] = (float)rand() / ((float)RAND_MAX + 1.0f);
    }

    ct_obj *t = ct_alloc_tensor(ndim, shape, (int)dtype, CUDA_G(current_device));
    if (!t) {
        efree(host);
        RETURN_THROWS();
    }

    cudaError_t err = cudaSuccess;
    if (dtype == CT_FP32) {
        err = cudaMemcpy(ct_data(t), host, (size_t)total * sizeof(float), cudaMemcpyHostToDevice);
    } else {
        void *staging = NULL;
        err = cudaMalloc(&staging, (size_t)total * sizeof(float));
        if (err == cudaSuccess) {
            err = cudaMemcpy(staging, host, (size_t)total * sizeof(float), cudaMemcpyHostToDevice);
            if (err == cudaSuccess) {
                err = ct_copy_cast(staging, ct_data(t), total, CT_FP32, (int)dtype, 0);
            }
            cudaFree(staging);
        }
    }
    efree(host);

    if (err != cudaSuccess) {
        ct_release(t);
        ct_throw_cuda(err, "CudaTensor::rand: copy failed");
        RETURN_THROWS();
    }
    ct_return_obj(return_value, t);
}

/* -------------------------------------------------------------------------
 * Introspection
 * ------------------------------------------------------------------------- */
ZEND_METHOD(CudaTensor, shape) {
    ZEND_PARSE_PARAMETERS_NONE();
    ct_obj *t = ct_obj_from_zval(getThis());
    array_init_size(return_value, t->ndim);
    for (int i = 0; i < t->ndim; i++) add_next_index_long(return_value, t->shape[i]);
}

ZEND_METHOD(CudaTensor, strides) {
    ZEND_PARSE_PARAMETERS_NONE();
    ct_obj *t = ct_obj_from_zval(getThis());
    array_init_size(return_value, t->ndim);
    for (int i = 0; i < t->ndim; i++) add_next_index_long(return_value, t->strides[i]);
}

ZEND_METHOD(CudaTensor, dtype) {
    ZEND_PARSE_PARAMETERS_NONE();
    RETURN_LONG(ct_obj_from_zval(getThis())->dtype);
}

ZEND_METHOD(CudaTensor, ndim) {
    ZEND_PARSE_PARAMETERS_NONE();
    RETURN_LONG(ct_obj_from_zval(getThis())->ndim);
}

ZEND_METHOD(CudaTensor, size) {
    ZEND_PARSE_PARAMETERS_NONE();
    ct_obj *t = ct_obj_from_zval(getThis());
    RETURN_LONG((zend_long)ct_numel(t->ndim, t->shape));
}

ZEND_METHOD(CudaTensor, nbytes) {
    ZEND_PARSE_PARAMETERS_NONE();
    ct_obj *t = ct_obj_from_zval(getThis());
    RETURN_LONG((zend_long)(ct_numel(t->ndim, t->shape) * (int64_t)ct_dtype_size(t->dtype)));
}

ZEND_METHOD(CudaTensor, device) {
    ZEND_PARSE_PARAMETERS_NONE();
    RETURN_LONG(ct_obj_from_zval(getThis())->storage->device_id);
}

ZEND_METHOD(CudaTensor, __toString) {
    ZEND_PARSE_PARAMETERS_NONE();
    ct_obj *t = ct_obj_from_zval(getThis());
    smart_str buf = {0};
    smart_str_appends(&buf, "CudaTensor(shape=[");
    for (int i = 0; i < t->ndim; i++) {
        if (i) smart_str_appends(&buf, ", ");
        smart_str_append_long(&buf, t->shape[i]);
    }
    smart_str_appends(&buf, "], dtype=");
    smart_str_appends(&buf, ct_dtype_name(t->dtype));
    smart_str_appends(&buf, ", device=");
    smart_str_append_long(&buf, t->storage->device_id);
    smart_str_appendc(&buf, ')');
    smart_str_0(&buf);
    RETURN_STR(buf.s);
}

ZEND_METHOD(CudaTensor, toArray) {
    ZEND_PARSE_PARAMETERS_NONE();
    ct_obj *t = ct_obj_from_zval(getThis());

    int64_t total;
    float *host = ct_to_host_f32(t, &total);
    if (!host) RETURN_THROWS();

    size_t pos = 0;
    ct_build_nested(host, t->ndim, t->shape, 0, &pos, return_value);
    efree(host);
}

/* -------------------------------------------------------------------------
 * Arithmetic
 * ------------------------------------------------------------------------- */
ZEND_METHOD(CudaTensor, add)  { ct_binary_op(INTERNAL_FUNCTION_PARAM_PASSTHRU, CT_OP_ADD); }
ZEND_METHOD(CudaTensor, sub)  { ct_binary_op(INTERNAL_FUNCTION_PARAM_PASSTHRU, CT_OP_SUB); }
ZEND_METHOD(CudaTensor, mul)  { ct_binary_op(INTERNAL_FUNCTION_PARAM_PASSTHRU, CT_OP_MUL); }
ZEND_METHOD(CudaTensor, div)  { ct_binary_op(INTERNAL_FUNCTION_PARAM_PASSTHRU, CT_OP_DIV); }

ZEND_METHOD(CudaTensor, add_) { ct_binary_op_inplace(INTERNAL_FUNCTION_PARAM_PASSTHRU, CT_OP_ADD); }
ZEND_METHOD(CudaTensor, sub_) { ct_binary_op_inplace(INTERNAL_FUNCTION_PARAM_PASSTHRU, CT_OP_SUB); }
ZEND_METHOD(CudaTensor, mul_) { ct_binary_op_inplace(INTERNAL_FUNCTION_PARAM_PASSTHRU, CT_OP_MUL); }
ZEND_METHOD(CudaTensor, div_) { ct_binary_op_inplace(INTERNAL_FUNCTION_PARAM_PASSTHRU, CT_OP_DIV); }

ZEND_METHOD(CudaTensor, matmul) {
    zval *other_zv;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_OBJECT_OF_CLASS(other_zv, cuda_tensor_ce)
    ZEND_PARSE_PARAMETERS_END();

    ct_obj *a = ct_obj_from_zval(getThis());
    ct_obj *b = ct_obj_from_zval(other_zv);

    if (a->ndim != 2 || b->ndim != 2) {
        ct_throw("CudaTensor::matmul: both tensors must be 2-D (batched matmul is not implemented yet)");
        RETURN_THROWS();
    }
    if (a->shape[1] != b->shape[0]) {
        ct_throw("CudaTensor::matmul: inner dimensions do not match");
        RETURN_THROWS();
    }
    if (a->dtype != b->dtype || (a->dtype != CT_FP32 && a->dtype != CT_FP64)) {
        ct_throw("CudaTensor::matmul: both tensors must share dtype fp32 or fp64");
        RETURN_THROWS();
    }
    if (a->storage->device_id != b->storage->device_id) {
        ct_throw("CudaTensor::matmul: tensors are on different devices");
        RETURN_THROWS();
    }

    ct_obj *ac = ct_make_contiguous(a);
    ct_obj *bc = ct_make_contiguous(b);
    if (!ac || !bc) {
        if (ac) ct_release(ac);
        if (bc) ct_release(bc);
        RETURN_THROWS();
    }

    int64_t m = a->shape[0];
    int64_t k = a->shape[1];
    int64_t n = b->shape[1];
    int64_t out_shape[2] = {m, n};

    ct_obj *out = ct_alloc_tensor(2, out_shape, a->dtype, a->storage->device_id);
    if (!out) {
        ct_release(ac);
        ct_release(bc);
        RETURN_THROWS();
    }

    cublasHandle_t handle = cuda_get_cublas_handle(a->storage->device_id);
    cublasStatus_t status;

    if (a->dtype == CT_FP32) {
        const float alpha = 1.0f, beta = 0.0f;
        /* Row-major C(m x n) = A(m x k) * B(k x n) via the transpose trick:
         * compute C^T = B^T * A^T in column-major terms. */
        status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             (int)n, (int)m, (int)k,
                             &alpha,
                             (const float *)ct_data(bc), (int)n,
                             (const float *)ct_data(ac), (int)k,
                             &beta,
                             (float *)ct_data(out), (int)n);
    } else {
        const double alpha = 1.0, beta = 0.0;
        status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             (int)n, (int)m, (int)k,
                             &alpha,
                             (const double *)ct_data(bc), (int)n,
                             (const double *)ct_data(ac), (int)k,
                             &beta,
                             (double *)ct_data(out), (int)n);
    }

    ct_release(ac);
    ct_release(bc);

    if (status != CUBLAS_STATUS_SUCCESS) {
        ct_release(out);
        ct_throw("CudaTensor::matmul: cuBLAS gemm failed");
        RETURN_THROWS();
    }
    cudaDeviceSynchronize();
    ct_return_obj(return_value, out);
}

/* -------------------------------------------------------------------------
 * Activations / math
 * ------------------------------------------------------------------------- */
ZEND_METHOD(CudaTensor, relu)    { ct_unary_op(INTERNAL_FUNCTION_PARAM_PASSTHRU, CT_UNARY_RELU); }
ZEND_METHOD(CudaTensor, sigmoid) { ct_unary_op(INTERNAL_FUNCTION_PARAM_PASSTHRU, CT_UNARY_SIGMOID); }
ZEND_METHOD(CudaTensor, tanh)    { ct_unary_op(INTERNAL_FUNCTION_PARAM_PASSTHRU, CT_UNARY_TANH); }
ZEND_METHOD(CudaTensor, exp)     { ct_unary_op(INTERNAL_FUNCTION_PARAM_PASSTHRU, CT_UNARY_EXP); }
ZEND_METHOD(CudaTensor, log)     { ct_unary_op(INTERNAL_FUNCTION_PARAM_PASSTHRU, CT_UNARY_LOG); }
ZEND_METHOD(CudaTensor, sqrt)    { ct_unary_op(INTERNAL_FUNCTION_PARAM_PASSTHRU, CT_UNARY_SQRT); }
ZEND_METHOD(CudaTensor, gelu)    { ct_unary_op(INTERNAL_FUNCTION_PARAM_PASSTHRU, CT_UNARY_GELU); }
ZEND_METHOD(CudaTensor, neg)     { ct_unary_op(INTERNAL_FUNCTION_PARAM_PASSTHRU, CT_UNARY_NEG); }

ZEND_METHOD(CudaTensor, softmax) {
    ZEND_PARSE_PARAMETERS_NONE();

    ct_obj *a = ct_obj_from_zval(getThis());
    if (a->ndim < 1) {
        ct_throw("CudaTensor::softmax: tensor must have at least one dimension");
        RETURN_THROWS();
    }

    ct_obj *ac = ct_make_contiguous(a);
    if (!ac) RETURN_THROWS();

    ct_obj *out = ct_alloc_tensor(a->ndim, a->shape, a->dtype, a->storage->device_id);
    if (!out) {
        ct_release(ac);
        RETURN_THROWS();
    }

    int64_t cols = a->shape[a->ndim - 1];
    int64_t total = ct_numel(a->ndim, a->shape);
    int64_t rows = total / cols;

    cudaError_t err = ct_softmax(ct_data(ac), ct_data(out), rows, cols, a->dtype, 0);
    ct_release(ac);

    if (err != cudaSuccess) {
        ct_release(out);
        ct_throw_cuda(err, "CudaTensor::softmax: kernel failed");
        RETURN_THROWS();
    }
    cudaDeviceSynchronize();
    ct_return_obj(return_value, out);
}

/* -------------------------------------------------------------------------
 * Reductions
 * ------------------------------------------------------------------------- */
ZEND_METHOD(CudaTensor, sum) { ct_reduce_op(INTERNAL_FUNCTION_PARAM_PASSTHRU, CT_REDUCE_SUM); }
ZEND_METHOD(CudaTensor, max) { ct_reduce_op(INTERNAL_FUNCTION_PARAM_PASSTHRU, CT_REDUCE_MAX); }
ZEND_METHOD(CudaTensor, min) { ct_reduce_op(INTERNAL_FUNCTION_PARAM_PASSTHRU, CT_REDUCE_MIN); }

ZEND_METHOD(CudaTensor, mean) {
    ZEND_PARSE_PARAMETERS_NONE();
    ct_obj *a = ct_obj_from_zval(getThis());

    ct_dims shape, strides;
    memset(&shape, 0, sizeof(shape));
    memset(&strides, 0, sizeof(strides));
    memcpy(shape.v, a->shape, sizeof(int64_t) * a->ndim);
    memcpy(strides.v, a->strides, sizeof(int64_t) * a->ndim);

    int64_t total = ct_numel(a->ndim, a->shape);
    if (total == 0) {
        ct_throw("CudaTensor::mean: empty tensor");
        RETURN_THROWS();
    }

    double result = 0.0;
    cudaError_t err = ct_reduce(CT_REDUCE_SUM, ct_data(a), strides, shape, a->ndim,
                                total, a->dtype, &result, 0);
    if (err != cudaSuccess) {
        ct_throw_cuda(err, "CudaTensor::mean: reduction failed");
        RETURN_THROWS();
    }
    RETURN_DOUBLE(result / (double)total);
}

/* -------------------------------------------------------------------------
 * Views
 * ------------------------------------------------------------------------- */
ZEND_METHOD(CudaTensor, reshape) {
    zval *shape_zv;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY(shape_zv)
    ZEND_PARSE_PARAMETERS_END();

    ct_obj *a = ct_obj_from_zval(getThis());

    int64_t shape[CT_MAX_DIMS];
    int ndim;
    if (ct_parse_shape(shape_zv, shape, &ndim) == FAILURE) {
        ct_throw("CudaTensor::reshape: invalid shape");
        RETURN_THROWS();
    }
    if (ct_numel(ndim, shape) != ct_numel(a->ndim, a->shape)) {
        ct_throw("CudaTensor::reshape: element count must not change");
        RETURN_THROWS();
    }
    if (!ct_is_contiguous(a)) {
        ct_throw("CudaTensor::reshape: tensor is not contiguous (call ->contiguous() first)");
        RETURN_THROWS();
    }

    ct_obj *view = ct_view_of(a);
    view->ndim = ndim;
    memset(view->shape, 0, sizeof(view->shape));
    memset(view->strides, 0, sizeof(view->strides));
    memcpy(view->shape, shape, sizeof(int64_t) * ndim);
    ct_default_strides(ndim, shape, view->strides);
    ct_return_obj(return_value, view);
}

ZEND_METHOD(CudaTensor, transpose) {
    zval *axes_zv = NULL;
    ZEND_PARSE_PARAMETERS_START(0, 1)
        Z_PARAM_OPTIONAL
        Z_PARAM_ARRAY_OR_NULL(axes_zv)
    ZEND_PARSE_PARAMETERS_END();

    ct_obj *a = ct_obj_from_zval(getThis());
    ct_obj *view = ct_view_of(a);

    if (!axes_zv) {
        /* Reverse all axes. */
        for (int i = 0; i < a->ndim; i++) {
            view->shape[i] = a->shape[a->ndim - 1 - i];
            view->strides[i] = a->strides[a->ndim - 1 - i];
        }
    } else {
        HashTable *ht = Z_ARRVAL_P(axes_zv);
        if (zend_hash_num_elements(ht) != (uint32_t)a->ndim) {
            ct_release(view);
            ct_throw("CudaTensor::transpose: axes must be a permutation of all dimensions");
            RETURN_THROWS();
        }
        zend_bool seen[CT_MAX_DIMS] = {0};
        int i = 0;
        zval *zv;
        ZEND_HASH_FOREACH_VAL(ht, zv) {
            zend_long ax = zval_get_long(zv);
            if (ax < 0 || ax >= a->ndim || seen[ax]) {
                ct_release(view);
                ct_throw("CudaTensor::transpose: axes must be a permutation of all dimensions");
                RETURN_THROWS();
            }
            seen[ax] = 1;
            view->shape[i] = a->shape[ax];
            view->strides[i] = a->strides[ax];
            i++;
        } ZEND_HASH_FOREACH_END();
    }
    ct_return_obj(return_value, view);
}

ZEND_METHOD(CudaTensor, slice) {
    zend_long dim, start;
    zend_long length = -1;
    ZEND_PARSE_PARAMETERS_START(2, 3)
        Z_PARAM_LONG(dim)
        Z_PARAM_LONG(start)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(length)
    ZEND_PARSE_PARAMETERS_END();

    ct_obj *a = ct_obj_from_zval(getThis());

    if (dim < 0 || dim >= a->ndim) {
        ct_throw("CudaTensor::slice: dimension out of range");
        RETURN_THROWS();
    }
    if (start < 0 || start >= a->shape[dim]) {
        ct_throw("CudaTensor::slice: start out of range");
        RETURN_THROWS();
    }
    if (length < 0) length = a->shape[dim] - start;
    if (start + length > a->shape[dim]) {
        ct_throw("CudaTensor::slice: length out of range");
        RETURN_THROWS();
    }

    ct_obj *view = ct_view_of(a);
    view->offset += start * a->strides[dim];
    view->shape[dim] = length;
    ct_return_obj(return_value, view);
}

ZEND_METHOD(CudaTensor, contiguous) {
    ZEND_PARSE_PARAMETERS_NONE();
    ct_obj *a = ct_obj_from_zval(getThis());
    ct_obj *out = ct_make_contiguous(a);
    if (!out) RETURN_THROWS();
    ct_return_obj(return_value, out);
}

/* -------------------------------------------------------------------------
 * Class registration
 * ------------------------------------------------------------------------- */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ct_void, 0, 0, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_ct___toString, 0, 0, IS_STRING, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_ct_fromArray, 0, 0, 1)
    ZEND_ARG_TYPE_INFO(0, data, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO(0, dtype, IS_LONG, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_ct_shape_dtype, 0, 0, 1)
    ZEND_ARG_TYPE_INFO(0, shape, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO(0, dtype, IS_LONG, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_ct_full, 0, 0, 2)
    ZEND_ARG_TYPE_INFO(0, shape, IS_ARRAY, 0)
    ZEND_ARG_TYPE_INFO(0, value, IS_DOUBLE, 0)
    ZEND_ARG_TYPE_INFO(0, dtype, IS_LONG, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_ct_operand, 0, 0, 1)
    ZEND_ARG_INFO(0, other)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_ct_tensor, 0, 0, 1)
    ZEND_ARG_OBJ_INFO(0, other, CudaTensor, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_ct_reshape, 0, 0, 1)
    ZEND_ARG_TYPE_INFO(0, shape, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_ct_transpose, 0, 0, 0)
    ZEND_ARG_TYPE_INFO(0, axes, IS_ARRAY, 1)
ZEND_END_ARG_INFO()

ZEND_BEGIN_ARG_INFO_EX(arginfo_ct_slice, 0, 0, 2)
    ZEND_ARG_TYPE_INFO(0, dim, IS_LONG, 0)
    ZEND_ARG_TYPE_INFO(0, start, IS_LONG, 0)
    ZEND_ARG_TYPE_INFO(0, length, IS_LONG, 1)
ZEND_END_ARG_INFO()

static const zend_function_entry ct_methods[] = {
    ZEND_ME(CudaTensor, fromArray, arginfo_ct_fromArray, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(CudaTensor, zeros, arginfo_ct_shape_dtype, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(CudaTensor, ones, arginfo_ct_shape_dtype, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(CudaTensor, full, arginfo_ct_full, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
    ZEND_ME(CudaTensor, rand, arginfo_ct_shape_dtype, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)

    ZEND_ME(CudaTensor, shape, arginfo_ct_void, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaTensor, strides, arginfo_ct_void, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaTensor, dtype, arginfo_ct_void, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaTensor, ndim, arginfo_ct_void, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaTensor, size, arginfo_ct_void, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaTensor, nbytes, arginfo_ct_void, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaTensor, device, arginfo_ct_void, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaTensor, toArray, arginfo_ct_void, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaTensor, __toString, arginfo_ct___toString, ZEND_ACC_PUBLIC)

    ZEND_ME(CudaTensor, add, arginfo_ct_operand, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaTensor, sub, arginfo_ct_operand, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaTensor, mul, arginfo_ct_operand, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaTensor, div, arginfo_ct_operand, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaTensor, add_, arginfo_ct_operand, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaTensor, sub_, arginfo_ct_operand, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaTensor, mul_, arginfo_ct_operand, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaTensor, div_, arginfo_ct_operand, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaTensor, matmul, arginfo_ct_tensor, ZEND_ACC_PUBLIC)

    ZEND_ME(CudaTensor, relu, arginfo_ct_void, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaTensor, sigmoid, arginfo_ct_void, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaTensor, tanh, arginfo_ct_void, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaTensor, exp, arginfo_ct_void, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaTensor, log, arginfo_ct_void, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaTensor, sqrt, arginfo_ct_void, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaTensor, gelu, arginfo_ct_void, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaTensor, neg, arginfo_ct_void, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaTensor, softmax, arginfo_ct_void, ZEND_ACC_PUBLIC)

    ZEND_ME(CudaTensor, sum, arginfo_ct_void, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaTensor, mean, arginfo_ct_void, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaTensor, max, arginfo_ct_void, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaTensor, min, arginfo_ct_void, ZEND_ACC_PUBLIC)

    ZEND_ME(CudaTensor, reshape, arginfo_ct_reshape, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaTensor, transpose, arginfo_ct_transpose, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaTensor, slice, arginfo_ct_slice, ZEND_ACC_PUBLIC)
    ZEND_ME(CudaTensor, contiguous, arginfo_ct_void, ZEND_ACC_PUBLIC)
    PHP_FE_END
};

void php_cuda_tensor_minit(void) {
    zend_class_entry ce;
    INIT_CLASS_ENTRY(ce, "CudaTensor", ct_methods);
    cuda_tensor_ce = zend_register_internal_class(&ce);
    cuda_tensor_ce->ce_flags |= ZEND_ACC_FINAL;
#ifdef ZEND_ACC_NO_DYNAMIC_PROPERTIES
    cuda_tensor_ce->ce_flags |= ZEND_ACC_NO_DYNAMIC_PROPERTIES;
#endif
    cuda_tensor_ce->create_object = ct_create_object;

    memcpy(&ct_handlers, zend_get_std_object_handlers(), sizeof(ct_handlers));
    ct_handlers.offset = XtOffsetOf(ct_obj, std);
    ct_handlers.free_obj = ct_free_object;
    ct_handlers.clone_obj = ct_clone_object;

    zend_declare_class_constant_long(cuda_tensor_ce, "FP32", sizeof("FP32") - 1, CT_FP32);
    zend_declare_class_constant_long(cuda_tensor_ce, "FP64", sizeof("FP64") - 1, CT_FP64);
    zend_declare_class_constant_long(cuda_tensor_ce, "INT32", sizeof("INT32") - 1, CT_INT32);
    zend_declare_class_constant_long(cuda_tensor_ce, "FP16", sizeof("FP16") - 1, CT_FP16);
    zend_declare_class_constant_long(cuda_tensor_ce, "BF16", sizeof("BF16") - 1, CT_BF16);
    zend_declare_class_constant_long(cuda_tensor_ce, "INT8", sizeof("INT8") - 1, CT_INT8);
}

void php_cuda_tensor_mshutdown(void) {
    /* Nothing global to release; storages die with their objects. */
}
