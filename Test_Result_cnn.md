# Test Result

---

## CNN Resnet18

### Flop

```json
{
    "model_type": "<class 'model.CNNHARModel.CNNHARModel'>",
    "flops": 4193486336.0,
    "params": 11446863.0,
    "run_time": 0.0031539199352264404,
    "test_result": [
        {
            "val_loss": 0.3282932937145233,
            "val_acc": 0.9083333611488342
        }
    ]
}
```

### Module summary

```txt
============================================================================================================================================================================================================================
Layer (type:depth-idx)                        Input Shape               Output Shape              Param #                   Param %                   Kernel Shape              Mult-Adds                 Trainable
============================================================================================================================================================================================================================
CNNHARModel                                   [1, 3, 336, 336]          [1, 15]                   --                             --                   --                        --                        False
├─Sequential: 1-1                             [1, 3, 336, 336]          [1, 512, 1, 1]            --                             --                   --                        --                        False
│    └─Conv2d: 2-1                            [1, 3, 336, 336]          [1, 64, 168, 168]         (9,408)                     0.08%                   [7, 7]                    265,531,392               False
│    └─BatchNorm2d: 2-2                       [1, 64, 168, 168]         [1, 64, 168, 168]         (128)                       0.00%                   --                        128                       False
│    └─ReLU: 2-3                              [1, 64, 168, 168]         [1, 64, 168, 168]         --                             --                   --                        --                        --
│    └─MaxPool2d: 2-4                         [1, 64, 168, 168]         [1, 64, 84, 84]           --                             --                   3                         --                        --
│    └─Sequential: 2-5                        [1, 64, 84, 84]           [1, 64, 84, 84]           --                             --                   --                        --                        False
│    │    └─BasicBlock: 3-1                   [1, 64, 84, 84]           [1, 64, 84, 84]           (73,984)                    0.65%                   --                        520,225,024               False
│    │    └─BasicBlock: 3-2                   [1, 64, 84, 84]           [1, 64, 84, 84]           (73,984)                    0.65%                   --                        520,225,024               False
│    └─Sequential: 2-6                        [1, 64, 84, 84]           [1, 128, 42, 42]          --                             --                   --                        --                        False
│    │    └─BasicBlock: 3-3                   [1, 64, 84, 84]           [1, 128, 42, 42]          (230,144)                   2.01%                   --                        404,620,032               False
│    │    └─BasicBlock: 3-4                   [1, 128, 42, 42]          [1, 128, 42, 42]          (295,424)                   2.58%                   --                        520,225,280               False
│    └─Sequential: 2-7                        [1, 128, 42, 42]          [1, 256, 21, 21]          --                             --                   --                        --                        False
│    │    └─BasicBlock: 3-5                   [1, 128, 42, 42]          [1, 256, 21, 21]          (919,040)                   8.03%                   --                        404,620,800               False
│    │    └─BasicBlock: 3-6                   [1, 256, 21, 21]          [1, 256, 21, 21]          (1,180,672)                10.31%                   --                        520,225,792               False
│    └─Sequential: 2-8                        [1, 256, 21, 21]          [1, 512, 11, 11]          --                             --                   --                        --                        False
│    │    └─BasicBlock: 3-7                   [1, 256, 21, 21]          [1, 512, 11, 11]          (3,673,088)                32.09%                   --                        444,075,008               False
│    │    └─BasicBlock: 3-8                   [1, 512, 11, 11]          [1, 512, 11, 11]          (4,720,640)                41.24%                   --                        570,951,680               False
│    └─AdaptiveAvgPool2d: 2-9                 [1, 512, 11, 11]          [1, 512, 1, 1]            --                             --                   --                        --                        --
├─Sequential: 1-2                             [1, 512]                  [1, 15]                   --                             --                   --                        --                        False
│    └─Dropout: 2-10                          [1, 512]                  [1, 512]                  --                             --                   --                        --                        --
│    └─Linear: 2-11                           [1, 512]                  [1, 512]                  (262,656)                   2.29%                   --                        262,656                   False
│    └─ReLU: 2-12                             [1, 512]                  [1, 512]                  --                             --                   --                        --                        --
│    └─Dropout: 2-13                          [1, 512]                  [1, 512]                  --                             --                   --                        --                        --
│    └─Linear: 2-14                           [1, 512]                  [1, 15]                   (7,695)                     0.07%                   --                        7,695                     False
============================================================================================================================================================================================================================
Total params: 11,446,863
Trainable params: 0
Non-trainable params: 11,446,863
Total mult-adds (Units.GIGABYTES): 4.17
============================================================================================================================================================================================================================
Input size (MB): 1.35
Forward/backward pass size (MB): 89.86
Params size (MB): 45.79
Estimated Total Size (MB): 137.00
============================================================================================================================================================================================================================
```

### Model Structure

```txt
CNNHARModel(
  (feature_extractor): Sequential(
    (0): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    (4): Sequential(
      (0): BasicBlock(
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (1): BasicBlock(
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (5): Sequential(
      (0): BasicBlock(
        (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (downsample): Sequential(
          (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): BasicBlock(
        (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (6): Sequential(
      (0): BasicBlock(
        (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (downsample): Sequential(
          (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): BasicBlock(
        (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (7): Sequential(
      (0): BasicBlock(
        (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (downsample): Sequential(
          (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): BasicBlock(
        (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (8): AdaptiveAvgPool2d(output_size=(1, 1))
  )
  (classifier): Sequential(
    (0): Dropout(p=0.3, inplace=False)
    (1): Linear(in_features=512, out_features=512, bias=True)
    (2): ReLU(inplace=True)
    (3): Dropout(p=0.3, inplace=False)
    (4): Linear(in_features=512, out_features=15, bias=True)
  )
)
```

---

## CNN Resnet34

### Flop

```json
{
    "model_type": "<class 'model.CNNHARModel.CNNHARModel'>",
    "flops": 8417342976.0,
    "params": 21555023.0,
    "run_time": 0.004223008155822754,
    "test_result": [
        {
            "val_loss": 0.26666003465652466,
            "val_acc": 0.9226190447807312
        }
    ]
}
```

### Module summary

```txt
============================================================================================================================================================================================================================
Layer (type:depth-idx)                        Input Shape               Output Shape              Param #                   Param %                   Kernel Shape              Mult-Adds                 Trainable
============================================================================================================================================================================================================================
CNNHARModel                                   [1, 3, 336, 336]          [1, 15]                   --                             --                   --                        --                        False
├─Sequential: 1-1                             [1, 3, 336, 336]          [1, 512, 1, 1]            --                             --                   --                        --                        False
│    └─Conv2d: 2-1                            [1, 3, 336, 336]          [1, 64, 168, 168]         (9,408)                     0.04%                   [7, 7]                    265,531,392               False
│    └─BatchNorm2d: 2-2                       [1, 64, 168, 168]         [1, 64, 168, 168]         (128)                       0.00%                   --                        128                       False
│    └─ReLU: 2-3                              [1, 64, 168, 168]         [1, 64, 168, 168]         --                             --                   --                        --                        --
│    └─MaxPool2d: 2-4                         [1, 64, 168, 168]         [1, 64, 84, 84]           --                             --                   3                         --                        --
│    └─Sequential: 2-5                        [1, 64, 84, 84]           [1, 64, 84, 84]           --                             --                   --                        --                        False
│    │    └─BasicBlock: 3-1                   [1, 64, 84, 84]           [1, 64, 84, 84]           (73,984)                    0.34%                   --                        520,225,024               False
│    │    └─BasicBlock: 3-2                   [1, 64, 84, 84]           [1, 64, 84, 84]           (73,984)                    0.34%                   --                        520,225,024               False
│    │    └─BasicBlock: 3-3                   [1, 64, 84, 84]           [1, 64, 84, 84]           (73,984)                    0.34%                   --                        520,225,024               False
│    └─Sequential: 2-6                        [1, 64, 84, 84]           [1, 128, 42, 42]          --                             --                   --                        --                        False
│    │    └─BasicBlock: 3-4                   [1, 64, 84, 84]           [1, 128, 42, 42]          (230,144)                   1.07%                   --                        404,620,032               False
│    │    └─BasicBlock: 3-5                   [1, 128, 42, 42]          [1, 128, 42, 42]          (295,424)                   1.37%                   --                        520,225,280               False
│    │    └─BasicBlock: 3-6                   [1, 128, 42, 42]          [1, 128, 42, 42]          (295,424)                   1.37%                   --                        520,225,280               False
│    │    └─BasicBlock: 3-7                   [1, 128, 42, 42]          [1, 128, 42, 42]          (295,424)                   1.37%                   --                        520,225,280               False
│    └─Sequential: 2-7                        [1, 128, 42, 42]          [1, 256, 21, 21]          --                             --                   --                        --                        False
│    │    └─BasicBlock: 3-8                   [1, 128, 42, 42]          [1, 256, 21, 21]          (919,040)                   4.26%                   --                        404,620,800               False
│    │    └─BasicBlock: 3-9                   [1, 256, 21, 21]          [1, 256, 21, 21]          (1,180,672)                 5.48%                   --                        520,225,792               False
│    │    └─BasicBlock: 3-10                  [1, 256, 21, 21]          [1, 256, 21, 21]          (1,180,672)                 5.48%                   --                        520,225,792               False
│    │    └─BasicBlock: 3-11                  [1, 256, 21, 21]          [1, 256, 21, 21]          (1,180,672)                 5.48%                   --                        520,225,792               False
│    │    └─BasicBlock: 3-12                  [1, 256, 21, 21]          [1, 256, 21, 21]          (1,180,672)                 5.48%                   --                        520,225,792               False
│    │    └─BasicBlock: 3-13                  [1, 256, 21, 21]          [1, 256, 21, 21]          (1,180,672)                 5.48%                   --                        520,225,792               False
│    └─Sequential: 2-8                        [1, 256, 21, 21]          [1, 512, 11, 11]          --                             --                   --                        --                        False
│    │    └─BasicBlock: 3-14                  [1, 256, 21, 21]          [1, 512, 11, 11]          (3,673,088)                17.04%                   --                        444,075,008               False
│    │    └─BasicBlock: 3-15                  [1, 512, 11, 11]          [1, 512, 11, 11]          (4,720,640)                21.90%                   --                        570,951,680               False
│    │    └─BasicBlock: 3-16                  [1, 512, 11, 11]          [1, 512, 11, 11]          (4,720,640)                21.90%                   --                        570,951,680               False
│    └─AdaptiveAvgPool2d: 2-9                 [1, 512, 11, 11]          [1, 512, 1, 1]            --                             --                   --                        --                        --
├─Sequential: 1-2                             [1, 512]                  [1, 15]                   --                             --                   --                        --                        False
│    └─Dropout: 2-10                          [1, 512]                  [1, 512]                  --                             --                   --                        --                        --
│    └─Linear: 2-11                           [1, 512]                  [1, 512]                  (262,656)                   1.22%                   --                        262,656                   False
│    └─ReLU: 2-12                             [1, 512]                  [1, 512]                  --                             --                   --                        --                        --
│    └─Dropout: 2-13                          [1, 512]                  [1, 512]                  --                             --                   --                        --                        --
│    └─Linear: 2-14                           [1, 512]                  [1, 15]                   (7,695)                     0.04%                   --                        7,695                     False
============================================================================================================================================================================================================================
Total params: 21,555,023
Trainable params: 0
Non-trainable params: 21,555,023
Total mult-adds (Units.GIGABYTES): 8.38
============================================================================================================================================================================================================================
Input size (MB): 1.35
Forward/backward pass size (MB): 135.19
Params size (MB): 86.22
Estimated Total Size (MB): 222.77
============================================================================================================================================================================================================================
```

### Model Structure

```txt
CNNHARModel(
  (feature_extractor): Sequential(
    (0): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    (4): Sequential(
      (0): BasicBlock(
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (1): BasicBlock(
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (2): BasicBlock(
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (5): Sequential(
      (0): BasicBlock(
        (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (downsample): Sequential(
          (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): BasicBlock(
        (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (2): BasicBlock(
        (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (3): BasicBlock(
        (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (6): Sequential(
      (0): BasicBlock(
        (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (downsample): Sequential(
          (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): BasicBlock(
        (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (2): BasicBlock(
        (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (3): BasicBlock(
        (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (4): BasicBlock(
        (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (5): BasicBlock(
        (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (7): Sequential(
      (0): BasicBlock(
        (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (downsample): Sequential(
          (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): BasicBlock(
        (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (2): BasicBlock(
        (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (8): AdaptiveAvgPool2d(output_size=(1, 1))
  )
  (classifier): Sequential(
    (0): Dropout(p=0.3, inplace=False)
    (1): Linear(in_features=512, out_features=512, bias=True)
    (2): ReLU(inplace=True)
    (3): Dropout(p=0.3, inplace=False)
    (4): Linear(in_features=512, out_features=15, bias=True)
  )
)
```

---

## CNN Mobilenet_v2

### Flop

```json
{
    "model_type": "<class 'model.CNNHARModel.CNNHARModel'>",
    "flops": 752432896.0,
    "params": 2887439.0,
    "run_time": 0.004930784225463867,
    "test_result": [
        {
            "val_loss": 0.46379002928733826,
            "val_acc": 0.8642857074737549
        }
    ]
}
```

### Module summary

```txt
======================================================================================================================================================================================================================================
Layer (type:depth-idx)                                  Input Shape               Output Shape              Param #                   Param %                   Kernel Shape              Mult-Adds                 Trainable
======================================================================================================================================================================================================================================
CNNHARModel                                             [1, 3, 336, 336]          [1, 15]                   --                             --                   --                        --                        False
├─Sequential: 1-1                                       [1, 3, 336, 336]          [1, 1280, 1, 1]           --                             --                   --                        --                        False
│    └─Sequential: 2-1                                  [1, 3, 336, 336]          [1, 1280, 11, 11]         --                             --                   --                        --                        False
│    │    └─Conv2dNormActivation: 3-1                   [1, 3, 336, 336]          [1, 32, 168, 168]         (928)                       0.03%                   --                        24,385,600                False
│    │    └─InvertedResidual: 3-2                       [1, 32, 168, 168]         [1, 16, 168, 168]         (896)                       0.03%                   --                        22,579,296                False
│    │    └─InvertedResidual: 3-3                       [1, 16, 168, 168]         [1, 24, 84, 84]           (5,136)                     0.18%                   --                        65,705,904                False
│    │    └─InvertedResidual: 3-4                       [1, 24, 84, 84]           [1, 24, 84, 84]           (8,832)                     0.31%                   --                        57,916,272                False
│    │    └─InvertedResidual: 3-5                       [1, 24, 84, 84]           [1, 32, 42, 42]           (10,000)                    0.35%                   --                        34,800,832                False
│    │    └─InvertedResidual: 3-6                       [1, 32, 42, 42]           [1, 32, 42, 42]           (14,848)                    0.51%                   --                        24,725,056                False
│    │    └─InvertedResidual: 3-7                       [1, 32, 42, 42]           [1, 32, 42, 42]           (14,848)                    0.51%                   --                        24,725,056                False
│    │    └─InvertedResidual: 3-8                       [1, 32, 42, 42]           [1, 64, 21, 21]           (21,056)                    0.73%                   --                        17,019,968                False
│    │    └─InvertedResidual: 3-9                       [1, 64, 21, 21]           [1, 64, 21, 21]           (54,272)                    1.88%                   --                        23,201,792                False
│    │    └─InvertedResidual: 3-10                      [1, 64, 21, 21]           [1, 64, 21, 21]           (54,272)                    1.88%                   --                        23,201,792                False
│    │    └─InvertedResidual: 3-11                      [1, 64, 21, 21]           [1, 64, 21, 21]           (54,272)                    1.88%                   --                        23,201,792                False
│    │    └─InvertedResidual: 3-12                      [1, 64, 21, 21]           [1, 96, 21, 21]           (66,624)                    2.31%                   --                        28,620,864                False
│    │    └─InvertedResidual: 3-13                      [1, 96, 21, 21]           [1, 96, 21, 21]           (118,272)                   4.10%                   --                        51,059,712                False
│    │    └─InvertedResidual: 3-14                      [1, 96, 21, 21]           [1, 96, 21, 21]           (118,272)                   4.10%                   --                        51,059,712                False
│    │    └─InvertedResidual: 3-15                      [1, 96, 21, 21]           [1, 160, 11, 11]          (155,264)                   5.38%                   --                        36,166,784                False
│    │    └─InvertedResidual: 3-16                      [1, 160, 11, 11]          [1, 160, 11, 11]          (320,000)                  11.08%                   --                        38,220,800                False
│    │    └─InvertedResidual: 3-17                      [1, 160, 11, 11]          [1, 160, 11, 11]          (320,000)                  11.08%                   --                        38,220,800                False
│    │    └─InvertedResidual: 3-18                      [1, 160, 11, 11]          [1, 320, 11, 11]          (473,920)                  16.41%                   --                        56,806,720                False
│    │    └─Conv2dNormActivation: 3-19                  [1, 320, 11, 11]          [1, 1280, 11, 11]         (412,160)                  14.27%                   --                        49,564,160                False
│    └─AdaptiveAvgPool2d: 2-2                           [1, 1280, 11, 11]         [1, 1280, 1, 1]           --                             --                   --                        --                        --
├─Sequential: 1-2                                       [1, 1280]                 [1, 15]                   --                             --                   --                        --                        False
│    └─Dropout: 2-3                                     [1, 1280]                 [1, 1280]                 --                             --                   --                        --                        --
│    └─Linear: 2-4                                      [1, 1280]                 [1, 512]                  (655,872)                  22.71%                   --                        655,872                   False
│    └─ReLU: 2-5                                        [1, 512]                  [1, 512]                  --                             --                   --                        --                        --
│    └─Dropout: 2-6                                     [1, 512]                  [1, 512]                  --                             --                   --                        --                        --
│    └─Linear: 2-7                                      [1, 512]                  [1, 15]                   (7,695)                     0.27%                   --                        7,695                     False
======================================================================================================================================================================================================================================
Total params: 2,887,439
Trainable params: 0
Non-trainable params: 2,887,439
Total mult-adds (Units.MEGABYTES): 691.85
======================================================================================================================================================================================================================================
Input size (MB): 1.35
Forward/backward pass size (MB): 241.86
Params size (MB): 11.55
Estimated Total Size (MB): 254.77
======================================================================================================================================================================================================================================
```

### Model Structure

```txt
CNNHARModel(
  (feature_extractor): Sequential(
    (0): Sequential(
      (0): Conv2dNormActivation(
        (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU6(inplace=True)
      )
      (1): InvertedResidual(
        (conv): Sequential(
          (0): Conv2dNormActivation(
            (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
            (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU6(inplace=True)
          )
          (1): Conv2d(32, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (2): InvertedResidual(
        (conv): Sequential(
          (0): Conv2dNormActivation(
            (0): Conv2d(16, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU6(inplace=True)
          )
          (1): Conv2dNormActivation(
            (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=96, bias=False)
            (1): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU6(inplace=True)
          )
          (2): Conv2d(96, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (3): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (3): InvertedResidual(
        (conv): Sequential(
          (0): Conv2dNormActivation(
            (0): Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU6(inplace=True)
          )
          (1): Conv2dNormActivation(
            (0): Conv2d(144, 144, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=144, bias=False)
            (1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU6(inplace=True)
          )
          (2): Conv2d(144, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (3): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (4): InvertedResidual(
        (conv): Sequential(
          (0): Conv2dNormActivation(
            (0): Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU6(inplace=True)
          )
          (1): Conv2dNormActivation(
            (0): Conv2d(144, 144, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=144, bias=False)
            (1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU6(inplace=True)
          )
          (2): Conv2d(144, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (3): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (5): InvertedResidual(
        (conv): Sequential(
          (0): Conv2dNormActivation(
            (0): Conv2d(32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU6(inplace=True)
          )
          (1): Conv2dNormActivation(
            (0): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192, bias=False)
            (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU6(inplace=True)
          )
          (2): Conv2d(192, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (3): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (6): InvertedResidual(
        (conv): Sequential(
          (0): Conv2dNormActivation(
            (0): Conv2d(32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU6(inplace=True)
          )
          (1): Conv2dNormActivation(
            (0): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192, bias=False)
            (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU6(inplace=True)
          )
          (2): Conv2d(192, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (3): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (7): InvertedResidual(
        (conv): Sequential(
          (0): Conv2dNormActivation(
            (0): Conv2d(32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU6(inplace=True)
          )
          (1): Conv2dNormActivation(
            (0): Conv2d(192, 192, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=192, bias=False)
            (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU6(inplace=True)
          )
          (2): Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (3): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (8): InvertedResidual(
        (conv): Sequential(
          (0): Conv2dNormActivation(
            (0): Conv2d(64, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU6(inplace=True)
          )
          (1): Conv2dNormActivation(
            (0): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384, bias=False)
            (1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU6(inplace=True)
          )
          (2): Conv2d(384, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (3): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (9): InvertedResidual(
        (conv): Sequential(
          (0): Conv2dNormActivation(
            (0): Conv2d(64, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU6(inplace=True)
          )
          (1): Conv2dNormActivation(
            (0): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384, bias=False)
            (1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU6(inplace=True)
          )
          (2): Conv2d(384, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (3): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (10): InvertedResidual(
        (conv): Sequential(
          (0): Conv2dNormActivation(
            (0): Conv2d(64, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU6(inplace=True)
          )
          (1): Conv2dNormActivation(
            (0): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384, bias=False)
            (1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU6(inplace=True)
          )
          (2): Conv2d(384, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (3): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (11): InvertedResidual(
        (conv): Sequential(
          (0): Conv2dNormActivation(
            (0): Conv2d(64, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU6(inplace=True)
          )
          (1): Conv2dNormActivation(
            (0): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384, bias=False)
            (1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU6(inplace=True)
          )
          (2): Conv2d(384, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (3): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (12): InvertedResidual(
        (conv): Sequential(
          (0): Conv2dNormActivation(
            (0): Conv2d(96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU6(inplace=True)
          )
          (1): Conv2dNormActivation(
            (0): Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=576, bias=False)
            (1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU6(inplace=True)
          )
          (2): Conv2d(576, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (3): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (13): InvertedResidual(
        (conv): Sequential(
          (0): Conv2dNormActivation(
            (0): Conv2d(96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU6(inplace=True)
          )
          (1): Conv2dNormActivation(
            (0): Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=576, bias=False)
            (1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU6(inplace=True)
          )
          (2): Conv2d(576, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (3): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (14): InvertedResidual(
        (conv): Sequential(
          (0): Conv2dNormActivation(
            (0): Conv2d(96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU6(inplace=True)
          )
          (1): Conv2dNormActivation(
            (0): Conv2d(576, 576, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=576, bias=False)
            (1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU6(inplace=True)
          )
          (2): Conv2d(576, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (3): BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (15): InvertedResidual(
        (conv): Sequential(
          (0): Conv2dNormActivation(
            (0): Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU6(inplace=True)
          )
          (1): Conv2dNormActivation(
            (0): Conv2d(960, 960, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=960, bias=False)
            (1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU6(inplace=True)
          )
          (2): Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (3): BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (16): InvertedResidual(
        (conv): Sequential(
          (0): Conv2dNormActivation(
            (0): Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU6(inplace=True)
          )
          (1): Conv2dNormActivation(
            (0): Conv2d(960, 960, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=960, bias=False)
            (1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU6(inplace=True)
          )
          (2): Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (3): BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (17): InvertedResidual(
        (conv): Sequential(
          (0): Conv2dNormActivation(
            (0): Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU6(inplace=True)
          )
          (1): Conv2dNormActivation(
            (0): Conv2d(960, 960, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=960, bias=False)
            (1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU6(inplace=True)
          )
          (2): Conv2d(960, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (3): BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (18): Conv2dNormActivation(
        (0): Conv2d(320, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(1280, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU6(inplace=True)
      )
    )
    (1): AdaptiveAvgPool2d(output_size=(1, 1))
  )
  (classifier): Sequential(
    (0): Dropout(p=0.3, inplace=False)
    (1): Linear(in_features=1280, out_features=512, bias=True)
    (2): ReLU(inplace=True)
    (3): Dropout(p=0.3, inplace=False)
    (4): Linear(in_features=512, out_features=15, bias=True)
  )
)
```
