#!/bin/bash
python -m torch.distributed.launch --nproc_per_node 1 \
  --master_port 12345 swin_quant_flow.py \
  --quantize \
  --only-calib \
  --cfg SwinTransformer/configs/swin/swin_tiny_patch4_window7_224.yaml \
  --resume weights/swin_tiny_patch4_window7_224.pth \
  --data-path /root/space/projects/data/imagenet_1k \
  --num-calib-batch 10 \
  --calib-batch-size 8
