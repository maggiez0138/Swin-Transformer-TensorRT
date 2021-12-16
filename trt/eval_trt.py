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

from config import get_config
from trt.engine import allocate_buffers, do_inference, image_class_accurate, load_tensorrt_engine
from data.build import build_dataset


def parse_option():
    parser = argparse.ArgumentParser('Evaluation script of Swin Transformer TensorRT engine', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', required=True, type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', default='../imagenet_1k', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', default='./weights/swin_tiny_patch4_window7_224.engine', help='TensorRT engine')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')

    args, unparsed = parser.parse_known_args()
    config = get_config(args)
    return args, config

def create_dataset_eval(config):
    dataset_val, _ = build_dataset(is_train=False, config=config)

    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=False,
        drop_last=False
    )
    return data_loader_val


def validate(data_loader_val, model_path, config):
    total_cnt = 0
    accurate_cnt = 0

    with load_tensorrt_engine(model_path) as engine:
        inputs, outputs, bindings, stream = allocate_buffers(engine)
        with engine.create_execution_context() as context:
            start = time.time()
            for image, target in data_loader_val:
                if len(image) == config.DATA.BATCH_SIZE:
                    # print('total_cnt: ', total_cnt)
                    total_cnt += len(image)
                    cur_image = image.numpy()
                    batch_images = cur_image
                    batch_labels = target.numpy()

                    outputs_shape, outputs_trt = do_inference(batch=batch_images, context=context,
                                                               bindings=bindings, inputs=inputs,
                                                               outputs=outputs, stream=stream)
                    assert (len(outputs_trt) == 1)
                    accurate_cnt += image_class_accurate(list(outputs_trt.values())[0], batch_labels)
            duration = time.time() - start

    print("Evaluation of TRT QAT model on {} images: {}, fps: {}".format(total_cnt,
                                                                         float(accurate_cnt) / float(total_cnt),
                                                                         float(total_cnt) / float(duration)))
    print('Duration: ', duration)


if __name__ == '__main__':
    _, config = parse_option()
    data_loader_val = create_dataset_eval(config)
    validate(data_loader_val, config.MODEL.RESUME, config)
