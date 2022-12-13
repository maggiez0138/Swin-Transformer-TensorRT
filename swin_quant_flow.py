# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import time
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from tqdm import tqdm
from copy import deepcopy
from prettytable import PrettyTable

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from SwinTransformer.config import get_config
from SwinTransformer.lr_scheduler import build_scheduler
from SwinTransformer.optimizer import build_optimizer
from SwinTransformer.logger import create_logger
from SwinTransformer.utils import load_checkpoint, save_checkpoint, get_grad_norm, reduce_tensor

from data import build_loader
from models import build_model
from export import export_onnx


try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None

try:
    from pytorch_quantization import nn as quant_nn
    from pytorch_quantization import calib
    from pytorch_quantization.tensor_quant import QuantDescriptor
except ImportError:
    raise ImportError(
        "pytorch-quantization is not installed. Install from "
        "https://github.com/NVIDIA/TensorRT/tree/master/tools/pytorch-quantization."
    )


class Knowledge_Distillation_Loss(torch.nn.Module):
    def __init__(self, scale, T = 3):
        super(Knowledge_Distillation_Loss, self).__init__()
        self.KLdiv = torch.nn.KLDivLoss()
        self.T = T
        self.scale = scale

    def get_knowledge_distillation_loss(self, output_student, output_teacher):
        loss_kl = self.KLdiv(torch.nn.functional.log_softmax(output_student / self.T, dim=1), torch.nn.functional.softmax(output_teacher / self.T, dim=1))

        loss = loss_kl
        return self.scale * loss


def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--amp-opt-level', type=str, default='O0', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')


    # settings for quantization and calibration
    parser.add_argument('--quantize', action='store_true', help='enable QAT wrapper')
    parser.add_argument('--only-calib', action='store_true', help='Perform calibration only, no QAT finetuing')
    parser.add_argument('--calib-batch-size', type=int,
                        default=8, help='calib batch size: default 64')
    parser.add_argument('--num-calib-batch', default=4, type=int,
                        help='Number of batches for calibration. 0 will disable calibration. (default: 4)')
    parser.add_argument('--calibrator', type=str, choices=["max", "histogram", "mse", "entropy"], default="histogram")
    parser.add_argument('--percentile', default=99.99, type=float, choices=[99.9, 99.99, 99.999, 99.9999],
                        help='percentile for PercentileCalibrator')
    parser.add_argument('--sensitivity', action="store_true", help="Build sensitivity profile")
    parser.add_argument("--accu-tolerance", type=float, default=0.925, help="used by test, for imagenet")
    parser.add_argument('--skip-layers', action="store_true", help='Skip some sensitivity layers')
    parser.add_argument('--dynamic', action='store_true', help='dynamic ONNX axes')
	
    parser.add_argument('--num-epochs', default=3, type=int,
                        help='Number of epochs to fine tune. 0 will disable fine tune. (default: 0)')
    parser.add_argument("--qat-lr", type=float, default=5e-7, help="learning rate for QAT.")
    parser.add_argument("--distill", action='store_true', help='Using distillation')
    parser.add_argument("--teacher", type=str, help='teacher model path')
    parser.add_argument('--distillation_loss_scale', type=float, default=10000., help="scale applied to distillation component of loss")

    # distributed training
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')
    # for acceleration
    parser.add_argument('--fused_window_process', action='store_true', help='Fused window shift & window partition, similar for reversed part.')


    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def main(config, args, logger):
    ## Step 1: create the calibration dataset
    dataset_train, dataset_val, data_loader_train, data_loader_calib, data_loader_val, mixup_fn = build_loader(config, args)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")

    ## Step1: Initialize calibration method
    if args.quantize:
        quant_desc_input = QuantDescriptor(calib_method=args.calibrator)
        quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
        quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)

    ## Step2: If --quantize enabled, will create the fake quantized model
    model = build_model(config, args.quantize)
    model.cuda()
    # PRINT the details of model (quantized, with TensorQuantizer inserted)
    # logger.info(str(model))

    optimizer = build_optimizer(config, model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
    model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    if config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    max_accuracy = 0.0

    if config.MODEL.RESUME:
        # max_accuracy = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
        msg = model_without_ddp.load_state_dict(checkpoint['model'] if 'model' in checkpoint.keys() else checkpoint, strict=False)
        logger.info(msg)
        del checkpoint

    # Calibrate the model
    with torch.no_grad():
        calibrate_model(
            model=model,
            model_name=config.MODEL.NAME,
            data_loader=data_loader_calib,
            num_calib_batch=args.num_calib_batch,
            calibrator=args.calibrator,
            hist_percentile=args.percentile,
            out_dir=config.OUTPUT,
            device=config.LOCAL_RANK)

        # Evaluate after calibration
        acc1_calibrated, acc5_calibrated, loss_calibrated = validate(config, data_loader_val, model)
        print('Evaluation after calibration: ', "{:.3f}, {:.3f}".format(acc1_calibrated, acc5_calibrated))

    if not args.only_calib:
        logger.info("Start training")
        start_time = time.time()
        #for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        teacher = None
        distillation_loss = None
        if args.distill:
            teacher = build_model(config)
            print("Loading teacher model...")
            teacher_ckpt = torch.load(args.teacher, map_location="cpu")
            if "model" in teacher_ckpt:
                teacher.load_state_dict(teacher_ckpt["model"], strict=False)
            else:
                teacher.load_state_dict(teacher_ckpt, strict=False)
            distillation_loss = Knowledge_Distillation_Loss(scale=args.distillation_loss_scale).cuda()
            teacher.cuda()
            teacher.eval()

        for epoch in range(args.num_epochs):
            data_loader_train.sampler.set_epoch(epoch)

            train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch,
                    mixup_fn, teacher, distillation_loss)
            if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
                    save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, logger)

            acc1_finetuned, acc5_finetuned, loss_finetuned = validate(config, data_loader_val, model)
            logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1_finetuned:.1f}%")
            max_accuracy = max(max_accuracy, acc1_finetuned)
            logger.info(f'Max accuracy: {max_accuracy:.2f}%')

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info('Training time {}'.format(total_time_str))

        # Print summary
        print("Accuracy summary:")
        table = PrettyTable(['Stage','Top1'])
        table.align['Stage'] = "l"
        table.add_row( [ 'Calibrated',  "{:.3f}, {:.3f}".format(acc1_calibrated, acc5_calibrated) ] )
        if config.NUM_FINETUNE_EPOCHS > 0:
            table.add_row( [ 'Finetuned',   "{:.3f}, {:.3f}".format(acc1_finetuned, acc5_finetuned) ] )
        print(table)

    export_onnx(model_without_ddp, config)

    return 0


def calibrate_model(model, model_name, data_loader, num_calib_batch, calibrator, hist_percentile, out_dir, device):
    """
        Feed data to the network and calibrate.
        Arguments:
            model: classification model
            model_name: name to use when creating state files
            data_loader: calibration data set
            num_calib_batch: amount of calibration passes to perform
            calibrator: type of calibration to use (max/histogram)
            hist_percentile: percentiles to be used for historgram calibration
            out_dir: dir to save state files in
    """

    if num_calib_batch > 0:
        print("Calibrating model")
        with torch.no_grad():
            collect_stats(model, data_loader, num_calib_batch, device)

        if calibrator == "max":
            compute_amax(model, method="max")
            calib_output = os.path.join(
                out_dir,
                F"{model_name}-max-{num_calib_batch*data_loader.batch_size}.pth")

            ckpt = {'model': deepcopy(model)}
            torch.save(ckpt, calib_output)
        elif calibrator in ["mse", "entropy"]:
            print(F"{calibrator} calibration")
            compute_amax(model, method=calibrator)

            calib_output = os.path.join(
                out_dir,
                F"{model_name}-{calibrator}-{num_calib_batch * data_loader.batch_size}.pth")

            ckpt = {'model': deepcopy(model)}
            torch.save(ckpt, calib_output)
        else:
            print(F"{args.percentile} percentile calibration")
            compute_amax(model, method="percentile", percentile=args.percentile)
            calib_output = os.path.join(
                out_dir,
                F"{model_name}-percentile-{args.percentile}-{num_calib_batch*data_loader.batch_size}.pth")

            ckpt = {'model': deepcopy(model)}
            torch.save(ckpt, calib_output)


def collect_stats(model, data_loader, num_batches, device):
    """Feed data to the network and collect statistics"""
    # Enable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    # Feed data to the network for collecting stats
    for i, (img, _) in tqdm(enumerate(data_loader), total=num_batches):
        img = img.to(device, non_blocking=True)
        # img = img.float()  # uint8 to fp16/32
        # img /= 255.0  # 0 - 255 to 0.0 - 1.0
        model(img)
        if i >= num_batches:
            break

    # Disable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()

def compute_amax(model, **kwargs):
    # Load calib result
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
            # print(F"{name:40}: {module}")
    model.cuda()

def train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, teacher, dis_loss):
    model.train()
    optimizer.zero_grad()
    max_accuracy = 0.0

    num_steps = len(data_loader_train)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, (samples, targets) in enumerate(data_loader_train):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        outputs = model(samples)
        loss = criterion(outputs, targets)
        if teacher:
            with torch.no_grad():
                teacher_outputs = teacher(samples)
            loss_t = dis_loss.get_knowledge_distillation_loss(outputs, teacher_outputs)
            loss = loss + loss_t
        optimizer.zero_grad()
        loss.backward()
        if config.TRAIN.CLIP_GRAD:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
        else:
            grad_norm = get_grad_norm(model.parameters())
        optimizer.step()
        # lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")
    

@torch.no_grad()
def validate(config, data_loader, model):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(images)

        # measure accuracy and record loss
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc5)
        loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg


if __name__ == '__main__':
    args, config = parse_option()

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = args.qat_lr * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.EPOCHS = args.num_epochs
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())

    main(config, args, logger)
