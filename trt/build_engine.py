"""
Evaluation script of Swin Transformer TensorRT engine
"""
import argparse
import torch
import time
import os
import sys
import numpy as np
sys.path.append('./')  # to run '$ python *.py' files in subdirectories

import tensorrt as trt
from tensorrt.tensorrt import IExecutionContext, Logger, Runtime
from trt.trt_utils import build_engine, save_engine


def parse_option():
    parser = argparse.ArgumentParser('TensorRT engine build script for Swin Transformer', add_help=False)
    parser.add_argument('--onnx-file', default='./weights/swin_tiny_patch4_window7_224.onnx', help='Onnx model file')
    parser.add_argument('--batch-size', default=16, type=int, help="batch size for single GPU")
    parser.add_argument('--trt-engine', default='./weights/swin_tiny_patch4_window7_224.engine', help='TensorRT engine')
    parser.add_argument('-v', '--verbose', action='store_true', help='enable verbose output (for debugging)')
    parser.add_argument('--mode', choices=['fp32', 'fp16', 'int8'], default='fp16')

    args = parser.parse_args()
    return args


def build_trt_engine():
    args = parse_option()

    trt_logger: Logger = trt.Logger(trt.Logger.VERBOSE) if args.verbose else trt.Logger()
    runtime: Runtime = trt.Runtime(trt_logger)

    engine = build_engine(
        runtime=runtime,
        onnx_file_path=args.onnx_file,
        logger=trt_logger,
        min_shape=(args.batch_size, 3, 224, 224),
        optimal_shape=(args.batch_size, 3, 224, 224),
        max_shape=(args.batch_size, 3, 224, 224),
        workspace_size=4<<30,
        mode=args.mode
    )

    save_engine(engine, args.trt_engine)


if __name__ == '__main__':
    build_trt_engine()


