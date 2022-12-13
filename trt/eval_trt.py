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

from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

from trt.trt_utils import allocate_buffers, do_inference, image_class_accurate, load_tensorrt_engine

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def parse_option():
    parser = argparse.ArgumentParser('Evaluation script of Swin Transformer TensorRT engine', add_help=False)
    parser.add_argument('--batch-size', required=True, type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', default='../imagenet_1k', type=str, help='path to dataset')
    parser.add_argument('--engine', default='./weights/swin_tiny_patch4_window7_224.engine', help='TensorRT engine')
    parser.add_argument('--img-size', type=int, default=224, help="input image size")

    args, unparsed = parser.parse_known_args()
    return args


def build_transform(img_size):
    t = []
    size = int((256 / 224) * img_size)
    t.append(
        transforms.Resize(size, interpolation=InterpolationMode.BICUBIC),
    )
    t.append(transforms.CenterCrop(img_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


def build_dataset(is_train, img_size, data_path, dataset_type='imagenet'):
    transform = build_transform(img_size)
    if dataset_type == 'imagenet':
        prefix = 'train' if is_train else 'val'
        root = os.path.join(data_path, prefix)
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif dataset_type == 'imagenet22K':
        raise NotImplementedError("Imagenet-22K will come soon.")
    else:
        raise NotImplementedError("We only support ImageNet Now.")

    return dataset, nb_classes


def build_loader(data_path, batch_size, img_size):
    dataset_val, _ = build_dataset(is_train=False, img_size=img_size, data_path=data_path)

    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=False
    )

    return dataset_val, data_loader_val


def validate(data_loader_val, model_path, batch_size):
    total_cnt = 0
    accurate_cnt = 0

    with load_tensorrt_engine(model_path) as engine:
        inputs, outputs, bindings, stream = allocate_buffers(engine)
        with engine.create_execution_context() as context:
            start = time.time()
            for image, target in data_loader_val:
                if len(image) == batch_size:
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

                    if total_cnt % 1000 == 0:
                        print("Processed {} images.".format(total_cnt))

            duration = time.time() - start

    print("Evaluation of TRT model on {} images: {}, fps: {}".format(total_cnt,
                                                                         float(accurate_cnt) / float(total_cnt),
                                                                         float(total_cnt) / float(duration)))
    print('Duration: ', duration)


if __name__ == '__main__':
    args = parse_option()
    _, data_loader_val = build_loader(args.data_path, args.batch_size, args.img_size)
    validate(data_loader_val, args.engine, args.batch_size)
