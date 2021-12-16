# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

from collections import OrderedDict
import logging
import traceback
import os
import sys
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

logger = logging.getLogger(__name__)


# Array of TensorRT loggers. We need to keep global references to
# the TensorRT loggers that we create to prevent them from being
# garbage collected as those are referenced from C++ code without
# Python knowing about it.

tensorrt_loggers = []


def _create_tensorrt_logger(verbose=False):
    """Create a TensorRT logger.

    Args:
        verbose (bool): whether to make the logger verbose.
    """
    if verbose:
        # trt_verbosity = trt.Logger.Severity.INFO
        trt_verbosity = trt.Logger.Severity.VERBOSE
    else:
        trt_verbosity = trt.Logger.Severity.WARNING
    tensorrt_logger = trt.Logger(trt_verbosity)
    tensorrt_loggers.append(tensorrt_logger)
    return tensorrt_logger


DEFAULT_MAX_WORKSPACE_SIZE = 1 << 30


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem, binding_name, shape=None):
        self.host = host_mem
        self.device = device_mem
        self.binding_name = binding_name
        self.shape = shape

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine):

    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem,device_mem, binding))
        else:
            output_shape = engine.get_binding_shape(binding)
            if len(output_shape)==3:
                dims = trt.DimsCHW(engine.get_binding_shape(binding))
                output_shape = (dims.c, dims.h, dims.w)
            elif len(output_shape)==2:
                dims = trt.Dims2(output_shape)
                output_shape = (dims[0], dims[1])
            outputs.append(HostDeviceMem(host_mem, device_mem, binding, output_shape))

    return inputs, outputs, bindings, stream


def do_inference(batch, context, bindings, inputs, outputs, stream):
    assert len(inputs) == 1
    inputs[0].host = np.ascontiguousarray(batch, dtype=np.float32)
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    stream.synchronize()

    outputs_dict = {}
    outputs_shape = {}
    for out in outputs:
        outputs_dict[out.binding_name] = np.reshape(out.host, out.shape)
        outputs_shape[out.binding_name] = out.shape

    return outputs_shape, outputs_dict

def load_tensorrt_engine(filename):
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    print('TRT model path: ', filename)
    with open(filename, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine

def image_class_accurate(pred, target):
    '''
    Return the number of accurate prediction.
    - pred: engine's output (batch_size, 1, class_num)
    - target: labels of sampl (batch_size, )
    '''
    pred = np.squeeze(pred)
    target = np.squeeze(target)
    pred_label = np.squeeze(np.argmax(pred, axis=-1))
    correct_cnt = np.sum(pred_label == target)
    # print('pred_label: ', pred_label, ' target: ', target)

    return correct_cnt
