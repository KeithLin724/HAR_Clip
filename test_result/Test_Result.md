# Test Result

---

## HAR Clip (Long Prompt)

### Flop

```json
{
    "model_type": "<class 'model.ClipHARModel.ClipHARModel'>",
    "flops": 193848616448.0,
    "params": 389960734.0,
    "run_time": 0.04079004669189453,
    "test_result": [
        {
            "val_loss": 0.11013045907020569,
            "val_acc": 0.9638888835906982
        }
    ]
}
```

### Module summary

```txt
================================================================================================================================================================================================================================================
Layer (type:depth-idx)                                            Input Shape               Output Shape              Param #                   Param %                   Kernel Shape              Mult-Adds                 Trainable
================================================================================================================================================================================================================================================
ClipHARModel                                                      [1, 3, 336, 336]          [1, 15]                   --                             --                   --                        --                        False
├─CLIPModel: 1-1                                                  --                        [1, 577, 1024]            1                           0.00%                   --                        --                        False
│    └─CLIPVisionTransformer: 2-1                                 --                        [1, 1024]                 --                             --                   --                        --                        False
│    │    └─CLIPVisionEmbeddings: 3-1                             [1, 3, 336, 336]          [1, 577, 1024]            (1,193,984)                 0.28%                   --                        347,407,360               False
│    │    └─LayerNorm: 3-2                                        [1, 577, 1024]            [1, 577, 1024]            (2,048)                     0.00%                   --                        2,048                     False
│    │    └─CLIPEncoder: 3-3                                      --                        [1, 577, 1024]            (302,309,376)              70.54%                   --                        302,309,376               False
│    │    └─LayerNorm: 3-4                                        [1, 1024]                 [1, 1024]                 (2,048)                     0.00%                   --                        2,048                     False
│    └─CLIPTextTransformer: 2-2                                   --                        [15, 768]                 --                             --                   --                        --                        False
│    │    └─CLIPTextEmbeddings: 3-5                               --                        [15, 15, 768]             (38,004,480)                8.87%                   --                        569,239,296               False
│    │    └─CLIPEncoder: 3-6                                      --                        [15, 15, 768]             (85,054,464)               19.85%                   --                        1,275,816,960             False
│    │    └─LayerNorm: 3-7                                        [15, 15, 768]             [15, 15, 768]             (1,536)                     0.00%                   --                        23,040                    False
│    └─Linear: 2-3                                                [1, 1024]                 [1, 768]                  (786,432)                   0.18%                   --                        786,432                   False
│    └─Linear: 2-4                                                [15, 768]                 [15, 768]                 (589,824)                   0.14%                   --                        8,847,360                 False
├─Linear: 1-2                                                     [1, 768]                  [1, 768]                  (589,824)                   0.14%                   --                        589,824                   False
├─Linear: 1-3                                                     [1, 768]                  [1, 15]                   (11,535)                    0.00%                   --                        11,535                    False
├─Linear: 1-4                                                     [1, 768]                  [1, 15]                   (11,535)                    0.00%                   --                        11,535                    False
================================================================================================================================================================================================================================================
Total params: 428,557,087
Trainable params: 0
Non-trainable params: 428,557,087
Total mult-adds (Units.GIGABYTES): 2.51
================================================================================================================================================================================================================================================
Input size (MB): 1.35
Forward/backward pass size (MB): 1447.49
Params size (MB): 1714.22
Estimated Total Size (MB): 3163.07
================================================================================================================================================================================================================================================
```

### Model Structure

```txt
ClipHARModel(
  (model): CLIPModel(
    (text_model): CLIPTextTransformer(
      (embeddings): CLIPTextEmbeddings(
        (token_embedding): Embedding(49408, 768)
        (position_embedding): Embedding(77, 768)
      )
      (encoder): CLIPEncoder(
        (layers): ModuleList(
          (0-11): 12 x CLIPEncoderLayer(
            (self_attn): CLIPSdpaAttention(
              (k_proj): Linear(in_features=768, out_features=768, bias=True)
              (v_proj): Linear(in_features=768, out_features=768, bias=True)
              (q_proj): Linear(in_features=768, out_features=768, bias=True)
              (out_proj): Linear(in_features=768, out_features=768, bias=True)
            )
            (layer_norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (mlp): CLIPMLP(
              (activation_fn): QuickGELUActivation()
              (fc1): Linear(in_features=768, out_features=3072, bias=True)
              (fc2): Linear(in_features=3072, out_features=768, bias=True)
            )
            (layer_norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          )
        )
      )
      (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    )
    (vision_model): CLIPVisionTransformer(
      (embeddings): CLIPVisionEmbeddings(
        (patch_embedding): Conv2d(3, 1024, kernel_size=(14, 14), stride=(14, 14), bias=False)
        (position_embedding): Embedding(577, 1024)
      )
      (pre_layrnorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
      (encoder): CLIPEncoder(
        (layers): ModuleList(
          (0-23): 24 x CLIPEncoderLayer(
            (self_attn): CLIPSdpaAttention(
              (k_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (v_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (q_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (out_proj): Linear(in_features=1024, out_features=1024, bias=True)
            )
            (layer_norm1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
            (mlp): CLIPMLP(
              (activation_fn): QuickGELUActivation()
              (fc1): Linear(in_features=1024, out_features=4096, bias=True)
              (fc2): Linear(in_features=4096, out_features=1024, bias=True)
            )
            (layer_norm2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
          )
        )
      )
      (post_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
    )
    (visual_projection): Linear(in_features=1024, out_features=768, bias=False)
    (text_projection): Linear(in_features=768, out_features=768, bias=False)
  )
  (weight_image_image): Linear(in_features=768, out_features=768, bias=False)
  (weight_image): Linear(in_features=768, out_features=15, bias=True)
  (weight_image_learnable): Linear(in_features=768, out_features=15, bias=True)
)
```

## HAR Clip (Short Prompt)

### Flop

```json
{
    "model_type": "<class 'model.ClipHARModel.ClipHARModel'>",
    "flops": 183647241728.0,
    "params": 389960734.0,
    "run_time": 0.03812326431274414,
    "test_result": [
        {
            "val_loss": 0.13923770189285278,
            "val_acc": 0.9559524059295654
        }
    ]
}
```

### Module summary

```txt
================================================================================================================================================================================================================================================
Layer (type:depth-idx)                                            Input Shape               Output Shape              Param #                   Param %                   Kernel Shape              Mult-Adds                 Trainable
================================================================================================================================================================================================================================================
ClipHARModel                                                      [1, 3, 336, 336]          [1, 15]                   --                             --                   --                        --                        False
├─CLIPModel: 1-1                                                  --                        [1, 577, 1024]            1                           0.00%                   --                        --                        False
│    └─CLIPVisionTransformer: 2-1                                 --                        [1, 1024]                 --                             --                   --                        --                        False
│    │    └─CLIPVisionEmbeddings: 3-1                             [1, 3, 336, 336]          [1, 577, 1024]            (1,193,984)                 0.28%                   --                        347,407,360               False
│    │    └─LayerNorm: 3-2                                        [1, 577, 1024]            [1, 577, 1024]            (2,048)                     0.00%                   --                        2,048                     False
│    │    └─CLIPEncoder: 3-3                                      --                        [1, 577, 1024]            (302,309,376)              70.54%                   --                        302,309,376               False
│    │    └─LayerNorm: 3-4                                        [1, 1024]                 [1, 1024]                 (2,048)                     0.00%                   --                        2,048                     False
│    └─CLIPTextTransformer: 2-2                                   --                        [15, 768]                 --                             --                   --                        --                        False
│    │    └─CLIPTextEmbeddings: 3-5                               --                        [15, 7, 768]              (38,004,480)                8.87%                   --                        569,239,296               False
│    │    └─CLIPEncoder: 3-6                                      --                        [15, 7, 768]              (85,054,464)               19.85%                   --                        1,275,816,960             False
│    │    └─LayerNorm: 3-7                                        [15, 7, 768]              [15, 7, 768]              (1,536)                     0.00%                   --                        23,040                    False
│    └─Linear: 2-3                                                [1, 1024]                 [1, 768]                  (786,432)                   0.18%                   --                        786,432                   False
│    └─Linear: 2-4                                                [15, 768]                 [15, 768]                 (589,824)                   0.14%                   --                        8,847,360                 False
├─Linear: 1-2                                                     [1, 768]                  [1, 768]                  (589,824)                   0.14%                   --                        589,824                   False
├─Linear: 1-3                                                     [1, 768]                  [1, 15]                   (11,535)                    0.00%                   --                        11,535                    False
├─Linear: 1-4                                                     [1, 768]                  [1, 15]                   (11,535)                    0.00%                   --                        11,535                    False
================================================================================================================================================================================================================================================
Total params: 428,557,087
Trainable params: 0
Non-trainable params: 428,557,087
Total mult-adds (Units.GIGABYTES): 2.51
================================================================================================================================================================================================================================================
Input size (MB): 1.35
Forward/backward pass size (MB): 1348.65
Params size (MB): 1714.22
Estimated Total Size (MB): 3064.22
================================================================================================================================================================================================================================================
```

### Model Structure

```txt
ClipHARModel(
  (model): CLIPModel(
    (text_model): CLIPTextTransformer(
      (embeddings): CLIPTextEmbeddings(
        (token_embedding): Embedding(49408, 768)
        (position_embedding): Embedding(77, 768)
      )
      (encoder): CLIPEncoder(
        (layers): ModuleList(
          (0-11): 12 x CLIPEncoderLayer(
            (self_attn): CLIPSdpaAttention(
              (k_proj): Linear(in_features=768, out_features=768, bias=True)
              (v_proj): Linear(in_features=768, out_features=768, bias=True)
              (q_proj): Linear(in_features=768, out_features=768, bias=True)
              (out_proj): Linear(in_features=768, out_features=768, bias=True)
            )
            (layer_norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (mlp): CLIPMLP(
              (activation_fn): QuickGELUActivation()
              (fc1): Linear(in_features=768, out_features=3072, bias=True)
              (fc2): Linear(in_features=3072, out_features=768, bias=True)
            )
            (layer_norm2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
          )
        )
      )
      (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    )
    (vision_model): CLIPVisionTransformer(
      (embeddings): CLIPVisionEmbeddings(
        (patch_embedding): Conv2d(3, 1024, kernel_size=(14, 14), stride=(14, 14), bias=False)
        (position_embedding): Embedding(577, 1024)
      )
      (pre_layrnorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
      (encoder): CLIPEncoder(
        (layers): ModuleList(
          (0-23): 24 x CLIPEncoderLayer(
            (self_attn): CLIPSdpaAttention(
              (k_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (v_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (q_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (out_proj): Linear(in_features=1024, out_features=1024, bias=True)
            )
            (layer_norm1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
            (mlp): CLIPMLP(
              (activation_fn): QuickGELUActivation()
              (fc1): Linear(in_features=1024, out_features=4096, bias=True)
              (fc2): Linear(in_features=4096, out_features=1024, bias=True)
            )
            (layer_norm2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
          )
        )
      )
      (post_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
    )
    (visual_projection): Linear(in_features=1024, out_features=768, bias=False)
    (text_projection): Linear(in_features=768, out_features=768, bias=False)
  )
  (weight_image_image): Linear(in_features=768, out_features=768, bias=False)
  (weight_image): Linear(in_features=768, out_features=15, bias=True)
  (weight_image_learnable): Linear(in_features=768, out_features=15, bias=True)
)
```
