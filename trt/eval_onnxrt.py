"""
Evaluation script of Swin Transformer TensorRT engine
"""
import argparse
import torch
import time
import sys
import numpy as np
import onnxruntime
sys.path.append('./')  # to run '$ python *.py' files in subdirectories


from trt.trt_utils import image_class_accurate
from SwinTransformer.config import get_config
from SwinTransformer.data.build import build_dataset


def get_input_shape(binding_dims):
    if len(binding_dims) == 4:
        return tuple(binding_dims[2:])
    elif len(binding_dims) == 3:
        return tuple(binding_dims[1:])
    else:
        raise ValueError('bad dims of binding %s' % (str(binding_dims)))


class Processor():
    def __init__(self, model):
        # load onnx engine
        self.ort_session = onnxruntime.InferenceSession(model)

        # get output name
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_names = []
        for i in range(len(self.ort_session.get_outputs())):
            output_name = self.ort_session.get_outputs()[i].name
            print("output name {}: ".format(i), output_name)
            output_shape = self.ort_session.get_outputs()[i].shape
            print("output shape {}: ".format(i), output_shape)
            self.output_names.append(output_name)

        self.input_shape = get_input_shape(self.ort_session.get_inputs()[0].shape)
        print('---self.input_shape: ', self.input_shape, self.ort_session.get_inputs()[0].shape)


    def inference(self, img):
        # forward model
        res = self.ort_session.run(self.output_names, {self.input_name: img})

        # Return only the host outputs.
        return res


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

    processor = Processor(model_path)

    start = time.time()
    for image, target in data_loader_val:
        if len(image) == config.DATA.BATCH_SIZE:
            print('total_cnt: ', total_cnt)
            total_cnt += len(image)
            cur_image = image.numpy()
            batch_images = cur_image
            batch_labels = target.numpy()

            outputs_onnxrt = processor.inference(batch_images)
            accurate_cnt += image_class_accurate(outputs_onnxrt, batch_labels)

        if total_cnt == config.DATA.BATCH_SIZE*20:
            break
    duration = time.time() - start

    print("Evaluation of TRT QAT model on {} images: {}, fps: {}".format(total_cnt,
                                                                         float(accurate_cnt) / float(total_cnt),
                                                                         float(total_cnt) / float(duration)))
    print('Duration: ', duration)


if __name__ == '__main__':
    _, config = parse_option()
    data_loader_val = create_dataset_eval(config)
    validate(data_loader_val, config.MODEL.RESUME, config)
