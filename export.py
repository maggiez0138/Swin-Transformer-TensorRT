# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import argparse
import torch

from SwinTransformer.config import get_config
from SwinTransformer.logger import create_logger
from SwinTransformer.utils import load_checkpoint

# Use the model definition modified
# 1) QAT support
# 2) Dynamic shape export support
from models import build_model

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None

try:
    from pytorch_quantization import nn as quant_nn
except ImportError:
    raise ImportError(
        "pytorch-quantization is not installed. Install from "
        "https://github.com/NVIDIA/TensorRT/tree/master/tools/pytorch-quantization."
    )


def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer export script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', default='../imagenet_1k', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', default='./weights/swin_tiny_patch4_window7_224.pth', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # If it is a QAT-wrapped model
    parser.add_argument('--quantize', action='store_true', help='enable QAT wrapper')

    # # settings for exporting onnx
    # parser.add_argument('--batch-size-onnx', type=int, help="batchsize when export the onnx model")

    # distributed training
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    # for acceleration
    parser.add_argument('--fused_window_process', action='store_true',
                        help='Fused window shift & window partition, similar for reversed part.')

    args, unparsed = parser.parse_known_args()
    config = get_config(args)
    return args, config


def export_onnx(model, config):
    # ONNX export
    try:
        model.eval()
        quant_nn.TensorQuantizer.use_fb_fake_quant = True  # We have to shift to pytorch's fake quant ops before exporting the model to ONNX

        import onnx
        dummy_input = torch.randn(1, 3, config.DATA.IMG_SIZE, config.DATA.IMG_SIZE, device='cuda')

        print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        f = config.MODEL.RESUME.replace('.pth', '.onnx')  # filename
        input_names = ["input_0"]
        output_names = ["output_0"]

        # Now with dynamic_axes, the output of TensorRT engine is wrong
        # So now we use fixed size
        dynamic_axes = {'input_0': {0: 'batch_size'}, 'output_0': {0: 'batch_size'}}
        torch.onnx.export(model, dummy_input, f, verbose=False, opset_version=13,
                          input_names=input_names,
                          output_names=output_names,
                          dynamic_axes=dynamic_axes,
                          do_constant_folding=True,
                          )
        print('ONNX export success, saved as %s' % f)
    except Exception as e:
        print('ONNX export failure: %s' % e)


def main(config, args):
    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config, args.quantize)
    model.cuda()
    logger.info(str(model))

    # import torchinfo
    # torchinfo.summary(model, (1, 3, 224, 224), depth=5)

    checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'] if 'model' in checkpoint.keys() else checkpoint,
                                            strict=False)
    logger.info(msg)
    del checkpoint

    export_onnx(model, config)


if __name__ == '__main__':
    args, config = parse_option()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=0, name=f"{config.MODEL.NAME}")

    # print config
    logger.info(config.dump())

    main(config, args)
