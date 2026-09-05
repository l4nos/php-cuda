#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "php.h"
#include "php_cuda.h"

/* -------------------------------------------------------------------------
 * Resource destructors (registered in cuda.c MINIT)
 * ------------------------------------------------------------------------- */
void php_cuda_stream_dtor(zend_resource *rsrc) {
    cuda_stream_resource *res = (cuda_stream_resource *)rsrc->ptr;
    if (!res) return;
    cudaSetDevice(res->device_id);
    cudaStreamDestroy(res->stream);
    efree(res);
}

void php_cuda_event_dtor(zend_resource *rsrc) {
    cuda_event_resource *res = (cuda_event_resource *)rsrc->ptr;
    if (!res) return;
    cudaSetDevice(res->device_id);
    cudaEventDestroy(res->start);
    cudaEventDestroy(res->stop);
    efree(res);
}

void php_cuda_graph_dtor(zend_resource *rsrc) {
    cuda_graph_resource *res = (cuda_graph_resource *)rsrc->ptr;
    if (!res) return;
    cudaSetDevice(res->device_id);
    cudaGraphExecDestroy(res->exec);
    efree(res);
}

/* -------------------------------------------------------------------------
 * Helpers
 * ------------------------------------------------------------------------- */
static cuda_stream_resource *fetch_stream(zval *zv) {
    return (cuda_stream_resource *)zend_fetch_resource(Z_RES_P(zv), "CUDA Stream", le_cuda_stream);
}

static cuda_event_resource *fetch_event(zval *zv) {
    return (cuda_event_resource *)zend_fetch_resource(Z_RES_P(zv), "CUDA Event", le_cuda_event);
}

static cuda_graph_resource *fetch_graph(zval *zv) {
    return (cuda_graph_resource *)zend_fetch_resource(Z_RES_P(zv), "CUDA Graph", le_cuda_graph);
}

/* Resolve an optional stream argument; NULL zval -> default stream (0). */
static cudaStream_t resolve_stream(zval *zv, int *device_id) {
    if (!zv) {
        if (device_id) *device_id = CUDA_G(current_device);
        return (cudaStream_t)0;
    }
    cuda_stream_resource *res = fetch_stream(zv);
    if (!res) return (cudaStream_t)-1; /* invalid */
    if (device_id) *device_id = res->device_id;
    return res->stream;
}

/* -------------------------------------------------------------------------
 * Streams
 * ------------------------------------------------------------------------- */
PHP_FUNCTION(cuda_stream_create) {
    ZEND_PARSE_PARAMETERS_NONE();

    CUDA_CHECK_RET(cuda_use_device(CUDA_G(current_device)));

    cuda_stream_resource *res = emalloc(sizeof(cuda_stream_resource));
    res->device_id = CUDA_G(current_device);
    CUDA_CHECK_RET(cudaStreamCreateWithFlags(&res->stream, cudaStreamNonBlocking));

    RETURN_RES(zend_register_resource(res, le_cuda_stream));
}

PHP_FUNCTION(cuda_stream_destroy) {
    zval *res;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_RESOURCE(res)
    ZEND_PARSE_PARAMETERS_END();

    if (!fetch_stream(res)) RETURN_FALSE;
    zend_list_close(Z_RES_P(res));
    RETURN_TRUE;
}

PHP_FUNCTION(cuda_stream_synchronize) {
    zval *res;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_RESOURCE(res)
    ZEND_PARSE_PARAMETERS_END();

    cuda_stream_resource *stream = fetch_stream(res);
    if (!stream) RETURN_FALSE;

    CUDA_CHECK_RET(cuda_use_device(stream->device_id));
    CUDA_CHECK_RET(cudaStreamSynchronize(stream->stream));
    RETURN_TRUE;
}

PHP_FUNCTION(cuda_stream_query) {
    zval *res;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_RESOURCE(res)
    ZEND_PARSE_PARAMETERS_END();

    cuda_stream_resource *stream = fetch_stream(res);
    if (!stream) RETURN_FALSE;

    CUDA_CHECK_RET(cuda_use_device(stream->device_id));
    cudaError_t err = cudaStreamQuery(stream->stream);
    if (err == cudaSuccess) {
        RETURN_TRUE;
    }
    if (err == cudaErrorNotReady) {
        RETURN_FALSE;
    }
    CUDA_CHECK_RET(err);
}

PHP_FUNCTION(cuda_stream_wait_event) {
    zval *stream_zv, *event_zv;
    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_RESOURCE(stream_zv)
        Z_PARAM_RESOURCE(event_zv)
    ZEND_PARSE_PARAMETERS_END();

    cuda_stream_resource *stream = fetch_stream(stream_zv);
    cuda_event_resource *event = fetch_event(event_zv);
    if (!stream || !event) RETURN_FALSE;

    CUDA_CHECK_RET(cuda_use_device(stream->device_id));
    /* Wait on the event's stop marker. */
    CUDA_CHECK_RET(cudaStreamWaitEvent(stream->stream, event->stop, 0));
    RETURN_TRUE;
}

/* -------------------------------------------------------------------------
 * Events (start/stop pair in one resource, for straightforward timing)
 * ------------------------------------------------------------------------- */
PHP_FUNCTION(cuda_event_create) {
    ZEND_PARSE_PARAMETERS_NONE();

    CUDA_CHECK_RET(cuda_use_device(CUDA_G(current_device)));

    cuda_event_resource *res = emalloc(sizeof(cuda_event_resource));
    res->device_id = CUDA_G(current_device);
    cudaError_t err = cudaEventCreate(&res->start);
    if (err == cudaSuccess) {
        err = cudaEventCreate(&res->stop);
    }
    if (err != cudaSuccess) {
        efree(res);
        CUDA_CHECK_RET(err);
    }

    RETURN_RES(zend_register_resource(res, le_cuda_event));
}

PHP_FUNCTION(cuda_event_destroy) {
    zval *res;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_RESOURCE(res)
    ZEND_PARSE_PARAMETERS_END();

    if (!fetch_event(res)) RETURN_FALSE;
    zend_list_close(Z_RES_P(res));
    RETURN_TRUE;
}

PHP_FUNCTION(cuda_event_record_start) {
    zval *event_zv, *stream_zv = NULL;
    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_RESOURCE(event_zv)
        Z_PARAM_OPTIONAL
        Z_PARAM_RESOURCE_OR_NULL(stream_zv)
    ZEND_PARSE_PARAMETERS_END();

    cuda_event_resource *event = fetch_event(event_zv);
    if (!event) RETURN_FALSE;

    int device_id;
    cudaStream_t stream = resolve_stream(stream_zv, &device_id);
    if (stream == (cudaStream_t)-1) RETURN_FALSE;

    CUDA_CHECK_RET(cuda_use_device(event->device_id));
    CUDA_CHECK_RET(cudaEventRecord(event->start, stream));
    RETURN_TRUE;
}

PHP_FUNCTION(cuda_event_record_stop) {
    zval *event_zv, *stream_zv = NULL;
    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_RESOURCE(event_zv)
        Z_PARAM_OPTIONAL
        Z_PARAM_RESOURCE_OR_NULL(stream_zv)
    ZEND_PARSE_PARAMETERS_END();

    cuda_event_resource *event = fetch_event(event_zv);
    if (!event) RETURN_FALSE;

    int device_id;
    cudaStream_t stream = resolve_stream(stream_zv, &device_id);
    if (stream == (cudaStream_t)-1) RETURN_FALSE;

    CUDA_CHECK_RET(cuda_use_device(event->device_id));
    CUDA_CHECK_RET(cudaEventRecord(event->stop, stream));
    RETURN_TRUE;
}

PHP_FUNCTION(cuda_event_elapsed_time) {
    zval *event_zv;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_RESOURCE(event_zv)
    ZEND_PARSE_PARAMETERS_END();

    cuda_event_resource *event = fetch_event(event_zv);
    if (!event) RETURN_FALSE;

    CUDA_CHECK_RET(cuda_use_device(event->device_id));
    CUDA_CHECK_RET(cudaEventSynchronize(event->stop));

    float ms = 0.0f;
    CUDA_CHECK_RET(cudaEventElapsedTime(&ms, event->start, event->stop));
    RETURN_DOUBLE((double)ms);
}

/* -------------------------------------------------------------------------
 * CUDA Graphs
 *
 * One capture may be active at a time (per process). If no stream is given
 * to cuda_graph_begin_capture(), an internal non-blocking stream is created
 * and reused for the matching end_capture.
 * ------------------------------------------------------------------------- */
static cudaStream_t php_cuda_capture_stream = NULL;
static int php_cuda_capture_device = -1;
static zend_bool php_cuda_capture_owned_stream = 0;

PHP_FUNCTION(cuda_graph_begin_capture) {
    zval *stream_zv = NULL;
    ZEND_PARSE_PARAMETERS_START(0, 1)
        Z_PARAM_OPTIONAL
        Z_PARAM_RESOURCE_OR_NULL(stream_zv)
    ZEND_PARSE_PARAMETERS_END();

    if (php_cuda_capture_stream) {
        cuda_report_error_msg("cuda_graph_begin_capture: a capture is already in progress");
        RETURN_FALSE;
    }

    int device_id;
    cudaStream_t stream;
    if (stream_zv) {
        stream = resolve_stream(stream_zv, &device_id);
        if (stream == (cudaStream_t)-1) RETURN_FALSE;
        php_cuda_capture_owned_stream = 0;
    } else {
        device_id = CUDA_G(current_device);
        CUDA_CHECK_RET(cuda_use_device(device_id));
        cudaError_t err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        if (err != cudaSuccess) {
            CUDA_CHECK_RET(err);
        }
        php_cuda_capture_owned_stream = 1;
    }

    CUDA_CHECK_RET(cuda_use_device(device_id));
    cudaError_t err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    if (err != cudaSuccess) {
        if (php_cuda_capture_owned_stream) cudaStreamDestroy(stream);
        CUDA_CHECK_RET(err);
    }

    php_cuda_capture_stream = stream;
    php_cuda_capture_device = device_id;
    RETURN_TRUE;
}

PHP_FUNCTION(cuda_graph_end_capture) {
    ZEND_PARSE_PARAMETERS_NONE();

    if (!php_cuda_capture_stream) {
        cuda_report_error_msg("cuda_graph_end_capture: no capture in progress");
        RETURN_FALSE;
    }

    cudaStream_t stream = php_cuda_capture_stream;
    int device_id = php_cuda_capture_device;
    php_cuda_capture_stream = NULL;
    php_cuda_capture_device = -1;

    CUDA_CHECK_RET(cuda_use_device(device_id));

    cudaGraph_t graph = NULL;
    cudaError_t err = cudaStreamEndCapture(stream, &graph);
    if (php_cuda_capture_owned_stream) {
        cudaStreamDestroy(stream);
        php_cuda_capture_owned_stream = 0;
    }
    if (err != cudaSuccess) {
        CUDA_CHECK_RET(err);
    }

    cuda_graph_resource *res = emalloc(sizeof(cuda_graph_resource));
    res->device_id = device_id;
    err = cudaGraphInstantiate(&res->exec, graph, NULL, NULL, 0);
    cudaGraphDestroy(graph);
    if (err != cudaSuccess) {
        efree(res);
        CUDA_CHECK_RET(err);
    }

    RETURN_RES(zend_register_resource(res, le_cuda_graph));
}

PHP_FUNCTION(cuda_graph_launch) {
    zval *graph_zv, *stream_zv = NULL;
    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_RESOURCE(graph_zv)
        Z_PARAM_OPTIONAL
        Z_PARAM_RESOURCE_OR_NULL(stream_zv)
    ZEND_PARSE_PARAMETERS_END();

    cuda_graph_resource *graph = fetch_graph(graph_zv);
    if (!graph) RETURN_FALSE;

    int device_id;
    cudaStream_t stream = resolve_stream(stream_zv, &device_id);
    if (stream == (cudaStream_t)-1) RETURN_FALSE;

    CUDA_CHECK_RET(cuda_use_device(graph->device_id));
    CUDA_CHECK_RET(cudaGraphLaunch(graph->exec, stream));
    RETURN_TRUE;
}

PHP_FUNCTION(cuda_graph_destroy) {
    zval *res;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_RESOURCE(res)
    ZEND_PARSE_PARAMETERS_END();

    if (!fetch_graph(res)) RETURN_FALSE;
    zend_list_close(Z_RES_P(res));
    RETURN_TRUE;
}
