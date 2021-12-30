# Swin Transformer

This project aims to explore the deployment of SwinTransformer based on TensorRT, including the test results of FP16 and INT8. 

## Introduction(Quoted from the Original Project )

**Swin Transformer** [original github repo](https://github.com/microsoft/Swin-Transformer/) (the name `Swin` stands for **S**hifted **win**dow) is initially described in [arxiv](https://arxiv.org/abs/2103.14030), which capably serves as a
general-purpose backbone for computer vision. It is basically a hierarchical Transformer whose representation is
computed with shifted windows. The shifted windowing scheme brings greater efficiency by limiting self-attention
computation to non-overlapping local windows while also allowing for cross-window connection.

## Setup ##

1. Please refer to the [Install](https://github.com/microsoft/Swin-Transformer/blob/main/get_started.md#install) session for conda environment build.  
2. Please refer to the [Data preparation](https://github.com/microsoft/Swin-Transformer/blob/main/get_started.md#data-preparation) session to prepare Imagenet-1K.
3. Install the TensorRT, now we choose the TensorRT 8.2 GA(8.2.1.8) as the test version.


## Code Structure ##  
Focus on the modifications and additions.
```
.
├── export.py                  # Export the PyTorch model to ONNX format
├── get_started.md            
├── main.py
├── models
│   ├── build.py
│   ├── __init__.py
│   ├── swin_mlp.py
│   └── swin_transformer.py    # Build the model, modified to export the onnx and build the TensorRT engine
├── README.md
├── trt                        # Directory for TensorRT's engine evaluation and visualization.
│   ├── engine.py
│   ├── eval_trt.py            # Evaluate the tensorRT engine's accuary.
│   ├── onnxrt_eval.py         # Run the onnx model, generate the results, just for debugging
├── utils.py
└── weights
```

## Export to ONNX and Build TensorRT Engine ##
You need to pay attention to the two modification below.  
1. Exporting the operator roll to ONNX opset version 9 is not supported. 
   A: Please refer to [torch/onnx/symbolic_opset9.py](torch/onnx/symbolic_opset9.py), add the support of exporting torch.roll.
   
2. Node (Concat_264) Op (Concat) [ShapeInferenceError] All inputs to Concat must have same rank.  
   A: Please refer to the modifications in `models/swin_transformer.py`. We use the input_resolution and window_size to compute the nW.
   ```css
      if mask is not None:
        nW = int(self.input_resolution[0]*self.input_resolution[1]/self.window_size[0]/self.window_size[1])
        #nW = mask.shape[0]
        #print('nW: ', nW)
        attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
    ```


## Accuray Test Results on ImageNet-1K Validation Dataset ##  
1. Download the `Swin-T` pretrained model from [Model Zoo](https://github.com/microsoft/Swin-Transformer/blob/main/get_started.md#regular-imagenet-1k-trained-models). 
Evaluate the accuracy of the Pytorch pretrained model.
    ```bash
    $ python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 main.py --eval --cfg configs/swin_tiny_patch4_window7_224.yaml --resume ./weights/swin_tiny_patch4_window7_224.pth --data-path ../imagenet_1k
    ```

2.  `export.py` exports a pytorch model to onnx format.
    ```bash
    $ python export.py --eval --cfg configs/swin_tiny_patch4_window7_224.yaml --resume ./weights/swin_tiny_patch4_window7_224.pth --data-path ../imagenet_1k --batch-size 16
    ```
    
3. Build the TensorRT engine using `trtexec`.  
    ```bash
    $ trtexec --onnx=./weights/swin_tiny_patch4_window7_224.onnx --buildOnly --verbose --saveEngine=./weights/swin_tiny_patch4_window7_224_batch16.engine --workspace=4096
    ```  
   
   Add the --fp16 or --best tag to build the corresponding fp16 or int8 model. Take fp16 as an example.  
   ```bash
   $ trtexec --onnx=./weights/swin_tiny_patch4_window7_224.onnx --buildOnly --verbose --fp16 --saveEngine=./weights/swin_tiny_patch4_window7_224_batch16_fp16.engine --workspace=4096
   ```  
   
   You can use the `trtexec` to test the throughput of the TensorRT engine.
   ```bash
   $ trtexec --loadEngine=./weights/swin_tiny_patch4_window7_224_batch16.engine
   ``` 

4.  ` trt/eval_trt.py` aims to evalute the accuracy of the TensorRT engine. 
   ```bash
   $ python trt/eval_trt.py --eval --cfg configs/swin_tiny_patch4_window7_224.yaml --resume ./weights/swin_tiny_patch4_window7_224_batch16.engine --data-path ../imagenet_1k --batch-size 16
   ```  

5. `trt/onnxrt_eval.py` aims to evalute the accuracy of the Onnx model, just for debug.
   ```bash
   $ python trt/onnxrt_eval.py --eval --cfg configs/swin_tiny_patch4_window7_224.yaml --resume ./weights/swin_tiny_patch4_window7_224.onnx --data-path ../imagenet_1k --batch-size 16
   ```  
   
| SwinTransformer(T4) | Acc@1 | Notes |
| :---: | :---: | :---: |
| PyTorch Pretrained Model |  81.160 |  |
| TensorRT Engine(FP32) | 81.156 |  |
| TensorRT Engine(FP16) | - | TensorRT 8.0.3.4: 81.156% vs TensorRT 8.2.1.8: 72.768% |

Notes: Need to check the FP16 overflow issue with TensorRT 8.2.1.8.

## Speed Test of TensorRT engine(T4) ##

| SwinTransformer(T4) | FP32 | FP16 | INT8 |
| :---: | :---: | :---: | :---: |
| batchsize=1 | 245.388 qps | 510.072 qps | 514.707 qps |
| batchsize=16 | 316.8624 qps | 804.112 qps | 804.1072 qps |
| batchsize=64 | 329.13984 qps | 833.4208 qps | 849.5168 qps |
| batchsize=256 | 331.9808 qps | 844.10752 qps | 840.33024 qps |

Analysis: Compared with FP16, INT8 does not speed up at present. And the current test results are expected.   
Attached the int8 and fp16 engine layer information with batchsize=128 on T4.  

Build with int8 precision:
```css
[12/04/2021-06:34:17] [V] [TRT] Engine Layer Information:
Layer(Reformat): Reformatting CopyNode for Input Tensor 0 to Conv_0, Tactic: 0, input_0[Float(128,3,224,224)] -> Reformatted Input Tensor 0 to Conv_0[Int8(128,3,224,224)]
Layer(CaskConvolution): Conv_0, Tactic: 1025026069226666066, Reformatted Input Tensor 0 to Conv_0[Int8(128,3,224,224)] -> 191[Int8(128,96,56,56)]
Layer(Reformat): Reformatting CopyNode for Input Tensor 0 to {ForeignNode[318...Transpose_2125 + Flatten_2127 + (Unnamed Layer* 4178) [Shuffle]]}, Tactic: 0, 191[Int8(128,96,56,56)] -> Reformatted Input Tensor 0 to {ForeignNode[318...Transpose_2125 + Flatten_2127 + (Unnamed Layer* 4178) [Shuffle]]}[Half(128,96,56,56)]
Layer(Myelin): {ForeignNode[318...Transpose_2125 + Flatten_2127 + (Unnamed Layer* 4178) [Shuffle]]}, Tactic: 0, Reformatted Input Tensor 0 to {ForeignNode[318...Transpose_2125 + Flatten_2127 + (Unnamed Layer* 4178) [Shuffle]]}[Half(128,96,56,56)] -> (Unnamed Layer* 4178) [Shuffle]_output[Half(128,768,1,1)]
Layer(CaskConvolution): Gemm_2128, Tactic: -1838109259315759592, (Unnamed Layer* 4178) [Shuffle]_output[Half(128,768,1,1)] -> (Unnamed Layer* 4179) [Fully Connected]_output[Half(128,1000,1,1)]
Layer(Reformat): Reformatting CopyNode for Input Tensor 0 to (Unnamed Layer* 4183) [Shuffle], Tactic: 0, (Unnamed Layer* 4179) [Fully Connected]_output[Half(128,1000,1,1)] -> Reformatted Input Tensor 0 to (Unnamed Layer* 4183) [Shuffle][Float(128,1000,1,1)]
Layer(NoOp): (Unnamed Layer* 4183) [Shuffle], Tactic: 0, Reformatted Input Tensor 0 to (Unnamed Layer* 4183) [Shuffle][Float(128,1000,1,1)] -> output_0[Float(128,1000)]
```  

Build with fp16 precision:
```css
[12/04/2021-06:44:31] [V] [TRT] Engine Layer Information:
Layer(Reformat): Reformatting CopyNode for Input Tensor 0 to Conv_0, Tactic: 0, input_0[Float(128,3,224,224)] -> Reformatted Input Tensor 0 to Conv_0[Half(128,3,224,224)]
Layer(CaskConvolution): Conv_0, Tactic: 1579845938601132607, Reformatted Input Tensor 0 to Conv_0[Half(128,3,224,224)] -> 191[Half(128,96,56,56)]
Layer(Myelin): {ForeignNode[318...(Unnamed Layer* 4183) [Shuffle]]}, Tactic: 0, 191[Half(128,96,56,56)] -> Reformatted Output Tensor 0 to {ForeignNode[318...(Unnamed Layer* 4183) [Shuffle]]}[Half(128,1000)]
Layer(Reformat): Reformatting CopyNode for Output Tensor 0 to {ForeignNode[318...(Unnamed Layer* 4183) [Shuffle]]}, Tactic: 0, Reformatted Output Tensor 0 to {ForeignNode[318...(Unnamed Layer* 4183) [Shuffle]]}[Half(128,1000)] -> output_0[Float(128,1000)]
```   

## Todo ##
After the FP16 issue solved, will do the QAT optimization.
