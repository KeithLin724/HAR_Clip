# Test Result

---

## LoRA Clip (torch.nn.Linear)

### Flop

```json
{
    "model_type": "<class 'model.ClipBaselineModel.ClipBaselineModel'>",
    "flops": "Skipped",
    "params": "Skipped",
    "run_time": 0.04087603378295898,
    "test_result": [
        {
            "val_acc": 0.9932539463043213,
            "val_loss": 0.023383110761642456
        }
    ]
}
```

### Module summary

```txt
==========================================================================================================================================================================================================================================================
Layer (type:depth-idx)                                                      Input Shape               Output Shape              Param #                   Param %                   Kernel Shape              Mult-Adds                 Trainable
==========================================================================================================================================================================================================================================================
ClipBaselineModel                                                           [1, 3, 336, 336]          [1, 577, 1024]            --                             --                   --                        --                        False
├─PeftModel: 1-1                                                            --                        [1, 577, 1024]            --                             --                   --                        --                        False
│    └─LoraModel: 2-1                                                       --                        --                        --                             --                   --                        --                        False
│    │    └─CLIPModel: 3-1                                                  --                        [1, 577, 1024]            (437,729,537)             100.00%                   --                        2,551,722,240             False
==========================================================================================================================================================================================================================================================
Total params: 437,729,537
Trainable params: 0
Non-trainable params: 437,729,537
Total mult-adds (Units.GIGABYTES): 2.55
==========================================================================================================================================================================================================================================================
Input size (MB): 1.35
Forward/backward pass size (MB): 2630.58
Params size (MB): 1750.91
Estimated Total Size (MB): 4382.85
==========================================================================================================================================================================================================================================================
```

### Model Structure

```txt
ClipBaselineModel(
  (model): PeftModel(
    (base_model): LoraModel(
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
                  (k_proj): lora.Linear(
                    (base_layer): Linear(in_features=768, out_features=768, bias=True)
                    (lora_dropout): ModuleDict(
                      (default): Dropout(p=0.1, inplace=False)
                    )
                    (lora_A): ModuleDict(
                      (default): Linear(in_features=768, out_features=16, bias=False)
                    )
                    (lora_B): ModuleDict(
                      (default): Linear(in_features=16, out_features=768, bias=False)
                    )
                    (lora_embedding_A): ParameterDict()
                    (lora_embedding_B): ParameterDict()
                    (lora_magnitude_vector): ModuleDict()
                  )
                  (v_proj): lora.Linear(
                    (base_layer): Linear(in_features=768, out_features=768, bias=True)
                    (lora_dropout): ModuleDict(
                      (default): Dropout(p=0.1, inplace=False)
                    )
                    (lora_A): ModuleDict(
                      (default): Linear(in_features=768, out_features=16, bias=False)
                    )
                    (lora_B): ModuleDict(
                      (default): Linear(in_features=16, out_features=768, bias=False)
                    )
                    (lora_embedding_A): ParameterDict()
                    (lora_embedding_B): ParameterDict()
                    (lora_magnitude_vector): ModuleDict()
                  )
                  (q_proj): lora.Linear(
                    (base_layer): Linear(in_features=768, out_features=768, bias=True)
                    (lora_dropout): ModuleDict(
                      (default): Dropout(p=0.1, inplace=False)
                    )
                    (lora_A): ModuleDict(
                      (default): Linear(in_features=768, out_features=16, bias=False)
                    )
                    (lora_B): ModuleDict(
                      (default): Linear(in_features=16, out_features=768, bias=False)
                    )
                    (lora_embedding_A): ParameterDict()
                    (lora_embedding_B): ParameterDict()
                    (lora_magnitude_vector): ModuleDict()
                  )
                  (out_proj): lora.Linear(
                    (base_layer): Linear(in_features=768, out_features=768, bias=True)
                    (lora_dropout): ModuleDict(
                      (default): Dropout(p=0.1, inplace=False)
                    )
                    (lora_A): ModuleDict(
                      (default): Linear(in_features=768, out_features=16, bias=False)
                    )
                    (lora_B): ModuleDict(
                      (default): Linear(in_features=16, out_features=768, bias=False)
                    )
                    (lora_embedding_A): ParameterDict()
                    (lora_embedding_B): ParameterDict()
                    (lora_magnitude_vector): ModuleDict()
                  )
                )
                (layer_norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (mlp): CLIPMLP(
                  (activation_fn): QuickGELUActivation()
                  (fc1): lora.Linear(
                    (base_layer): Linear(in_features=768, out_features=3072, bias=True)
                    (lora_dropout): ModuleDict(
                      (default): Dropout(p=0.1, inplace=False)
                    )
                    (lora_A): ModuleDict(
                      (default): Linear(in_features=768, out_features=16, bias=False)
                    )
                    (lora_B): ModuleDict(
                      (default): Linear(in_features=16, out_features=3072, bias=False)
                    )
                    (lora_embedding_A): ParameterDict()
                    (lora_embedding_B): ParameterDict()
                    (lora_magnitude_vector): ModuleDict()
                  )
                  (fc2): lora.Linear(
                    (base_layer): Linear(in_features=3072, out_features=768, bias=True)
                    (lora_dropout): ModuleDict(
                      (default): Dropout(p=0.1, inplace=False)
                    )
                    (lora_A): ModuleDict(
                      (default): Linear(in_features=3072, out_features=16, bias=False)
                    )
                    (lora_B): ModuleDict(
                      (default): Linear(in_features=16, out_features=768, bias=False)
                    )
                    (lora_embedding_A): ParameterDict()
                    (lora_embedding_B): ParameterDict()
                    (lora_magnitude_vector): ModuleDict()
                  )
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
                  (k_proj): lora.Linear(
                    (base_layer): Linear(in_features=1024, out_features=1024, bias=True)
                    (lora_dropout): ModuleDict(
                      (default): Dropout(p=0.1, inplace=False)
                    )
                    (lora_A): ModuleDict(
                      (default): Linear(in_features=1024, out_features=16, bias=False)
                    )
                    (lora_B): ModuleDict(
                      (default): Linear(in_features=16, out_features=1024, bias=False)
                    )
                    (lora_embedding_A): ParameterDict()
                    (lora_embedding_B): ParameterDict()
                    (lora_magnitude_vector): ModuleDict()
                  )
                  (v_proj): lora.Linear(
                    (base_layer): Linear(in_features=1024, out_features=1024, bias=True)
                    (lora_dropout): ModuleDict(
                      (default): Dropout(p=0.1, inplace=False)
                    )
                    (lora_A): ModuleDict(
                      (default): Linear(in_features=1024, out_features=16, bias=False)
                    )
                    (lora_B): ModuleDict(
                      (default): Linear(in_features=16, out_features=1024, bias=False)
                    )
                    (lora_embedding_A): ParameterDict()
                    (lora_embedding_B): ParameterDict()
                    (lora_magnitude_vector): ModuleDict()
                  )
                  (q_proj): lora.Linear(
                    (base_layer): Linear(in_features=1024, out_features=1024, bias=True)
                    (lora_dropout): ModuleDict(
                      (default): Dropout(p=0.1, inplace=False)
                    )
                    (lora_A): ModuleDict(
                      (default): Linear(in_features=1024, out_features=16, bias=False)
                    )
                    (lora_B): ModuleDict(
                      (default): Linear(in_features=16, out_features=1024, bias=False)
                    )
                    (lora_embedding_A): ParameterDict()
                    (lora_embedding_B): ParameterDict()
                    (lora_magnitude_vector): ModuleDict()
                  )
                  (out_proj): lora.Linear(
                    (base_layer): Linear(in_features=1024, out_features=1024, bias=True)
                    (lora_dropout): ModuleDict(
                      (default): Dropout(p=0.1, inplace=False)
                    )
                    (lora_A): ModuleDict(
                      (default): Linear(in_features=1024, out_features=16, bias=False)
                    )
                    (lora_B): ModuleDict(
                      (default): Linear(in_features=16, out_features=1024, bias=False)
                    )
                    (lora_embedding_A): ParameterDict()
                    (lora_embedding_B): ParameterDict()
                    (lora_magnitude_vector): ModuleDict()
                  )
                )
                (layer_norm1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
                (mlp): CLIPMLP(
                  (activation_fn): QuickGELUActivation()
                  (fc1): lora.Linear(
                    (base_layer): Linear(in_features=1024, out_features=4096, bias=True)
                    (lora_dropout): ModuleDict(
                      (default): Dropout(p=0.1, inplace=False)
                    )
                    (lora_A): ModuleDict(
                      (default): Linear(in_features=1024, out_features=16, bias=False)
                    )
                    (lora_B): ModuleDict(
                      (default): Linear(in_features=16, out_features=4096, bias=False)
                    )
                    (lora_embedding_A): ParameterDict()
                    (lora_embedding_B): ParameterDict()
                    (lora_magnitude_vector): ModuleDict()
                  )
                  (fc2): lora.Linear(
                    (base_layer): Linear(in_features=4096, out_features=1024, bias=True)
                    (lora_dropout): ModuleDict(
                      (default): Dropout(p=0.1, inplace=False)
                    )
                    (lora_A): ModuleDict(
                      (default): Linear(in_features=4096, out_features=16, bias=False)
                    )
                    (lora_B): ModuleDict(
                      (default): Linear(in_features=16, out_features=1024, bias=False)
                    )
                    (lora_embedding_A): ParameterDict()
                    (lora_embedding_B): ParameterDict()
                    (lora_magnitude_vector): ModuleDict()
                  )
                )
                (layer_norm2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              )
            )
          )
          (post_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        )
        (visual_projection): lora.Linear(
          (base_layer): Linear(in_features=1024, out_features=768, bias=False)
          (lora_dropout): ModuleDict(
            (default): Dropout(p=0.1, inplace=False)
          )
          (lora_A): ModuleDict(
            (default): Linear(in_features=1024, out_features=16, bias=False)
          )
          (lora_B): ModuleDict(
            (default): Linear(in_features=16, out_features=768, bias=False)
          )
          (lora_embedding_A): ParameterDict()
          (lora_embedding_B): ParameterDict()
          (lora_magnitude_vector): ModuleDict()
        )
        (text_projection): lora.Linear(
          (base_layer): Linear(in_features=768, out_features=768, bias=False)
          (lora_dropout): ModuleDict(
            (default): Dropout(p=0.1, inplace=False)
          )
          (lora_A): ModuleDict(
            (default): Linear(in_features=768, out_features=16, bias=False)
          )
          (lora_B): ModuleDict(
            (default): Linear(in_features=16, out_features=768, bias=False)
          )
          (lora_embedding_A): ParameterDict()
          (lora_embedding_B): ParameterDict()
          (lora_magnitude_vector): ModuleDict()
        )
      )
    )
  )
)
```

## LoRA Clip (torch.nn.Linear Text encoder)

### Flop
```json 
{
    "model_type": "<class 'model.ClipBaselineModel.ClipBaselineModel'>",
    "flops": "Skipped",
    "params": "Skipped",
    "run_time": 0.031307775497436525,
    "test_result": [
        {
            "val_acc": 0.9488095045089722,
            "val_loss": 0.16574490070343018
        }
    ]
}
```


### Module summary
```txt
==========================================================================================================================================================================================================================================================
Layer (type:depth-idx)                                                      Input Shape               Output Shape              Param #                   Param %                   Kernel Shape              Mult-Adds                 Trainable
==========================================================================================================================================================================================================================================================
ClipBaselineModel                                                           [1, 3, 336, 336]          [1, 577, 1024]            --                             --                   --                        --                        False
├─PeftModel: 1-1                                                            --                        [1, 577, 1024]            --                             --                   --                        --                        False
│    └─LoraModel: 2-1                                                       --                        --                        --                             --                   --                        --                        False
│    │    └─CLIPModel: 3-1                                                  --                        [1, 577, 1024]            (430,598,401)             100.00%                   --                        2,544,247,040             False
==========================================================================================================================================================================================================================================================
Total params: 430,598,401
Trainable params: 0
Non-trainable params: 430,598,401
Total mult-adds (Units.GIGABYTES): 2.54
==========================================================================================================================================================================================================================================================
Input size (MB): 1.35
Forward/backward pass size (MB): 1598.86
Params size (MB): 1722.39
Estimated Total Size (MB): 3322.60
==========================================================================================================================================================================================================================================================
```


### Model Structure
```txt
ClipBaselineModel(
  (model): PeftModel(
    (base_model): LoraModel(
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
                  (k_proj): lora.Linear(
                    (base_layer): Linear(in_features=768, out_features=768, bias=True)
                    (lora_dropout): ModuleDict(
                      (default): Dropout(p=0.1, inplace=False)
                    )
                    (lora_A): ModuleDict(
                      (default): Linear(in_features=768, out_features=16, bias=False)
                    )
                    (lora_B): ModuleDict(
                      (default): Linear(in_features=16, out_features=768, bias=False)
                    )
                    (lora_embedding_A): ParameterDict()
                    (lora_embedding_B): ParameterDict()
                    (lora_magnitude_vector): ModuleDict()
                  )
                  (v_proj): lora.Linear(
                    (base_layer): Linear(in_features=768, out_features=768, bias=True)
                    (lora_dropout): ModuleDict(
                      (default): Dropout(p=0.1, inplace=False)
                    )
                    (lora_A): ModuleDict(
                      (default): Linear(in_features=768, out_features=16, bias=False)
                    )
                    (lora_B): ModuleDict(
                      (default): Linear(in_features=16, out_features=768, bias=False)
                    )
                    (lora_embedding_A): ParameterDict()
                    (lora_embedding_B): ParameterDict()
                    (lora_magnitude_vector): ModuleDict()
                  )
                  (q_proj): lora.Linear(
                    (base_layer): Linear(in_features=768, out_features=768, bias=True)
                    (lora_dropout): ModuleDict(
                      (default): Dropout(p=0.1, inplace=False)
                    )
                    (lora_A): ModuleDict(
                      (default): Linear(in_features=768, out_features=16, bias=False)
                    )
                    (lora_B): ModuleDict(
                      (default): Linear(in_features=16, out_features=768, bias=False)
                    )
                    (lora_embedding_A): ParameterDict()
                    (lora_embedding_B): ParameterDict()
                    (lora_magnitude_vector): ModuleDict()
                  )
                  (out_proj): lora.Linear(
                    (base_layer): Linear(in_features=768, out_features=768, bias=True)
                    (lora_dropout): ModuleDict(
                      (default): Dropout(p=0.1, inplace=False)
                    )
                    (lora_A): ModuleDict(
                      (default): Linear(in_features=768, out_features=16, bias=False)
                    )
                    (lora_B): ModuleDict(
                      (default): Linear(in_features=16, out_features=768, bias=False)
                    )
                    (lora_embedding_A): ParameterDict()
                    (lora_embedding_B): ParameterDict()
                    (lora_magnitude_vector): ModuleDict()
                  )
                )
                (layer_norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (mlp): CLIPMLP(
                  (activation_fn): QuickGELUActivation()
                  (fc1): lora.Linear(
                    (base_layer): Linear(in_features=768, out_features=3072, bias=True)
                    (lora_dropout): ModuleDict(
                      (default): Dropout(p=0.1, inplace=False)
                    )
                    (lora_A): ModuleDict(
                      (default): Linear(in_features=768, out_features=16, bias=False)
                    )
                    (lora_B): ModuleDict(
                      (default): Linear(in_features=16, out_features=3072, bias=False)
                    )
                    (lora_embedding_A): ParameterDict()
                    (lora_embedding_B): ParameterDict()
                    (lora_magnitude_vector): ModuleDict()
                  )
                  (fc2): lora.Linear(
                    (base_layer): Linear(in_features=3072, out_features=768, bias=True)
                    (lora_dropout): ModuleDict(
                      (default): Dropout(p=0.1, inplace=False)
                    )
                    (lora_A): ModuleDict(
                      (default): Linear(in_features=3072, out_features=16, bias=False)
                    )
                    (lora_B): ModuleDict(
                      (default): Linear(in_features=16, out_features=768, bias=False)
                    )
                    (lora_embedding_A): ParameterDict()
                    (lora_embedding_B): ParameterDict()
                    (lora_magnitude_vector): ModuleDict()
                  )
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
    )
  )
)
```


## LoRA Clip (torch.nn.Linear Vision Encoder)

### Flop
```json
{
    "model_type": "<class 'model.ClipBaselineModel.ClipBaselineModel'>",
    "flops": "Skipped",
    "params": "Skipped",
    "run_time": 0.03389440155029297,
    "test_result": [
        {
            "val_acc": 0.9932539463043213,
            "val_loss": 0.026230541989207268
        }
    ]
}
```

### Module summary
```txt
==========================================================================================================================================================================================================================================================
Layer (type:depth-idx)                                                      Input Shape               Output Shape              Param #                   Param %                   Kernel Shape              Mult-Adds                 Trainable
==========================================================================================================================================================================================================================================================
ClipBaselineModel                                                           [1, 3, 336, 336]          [1, 577, 1024]            --                             --                   --                        --                        False
├─PeftModel: 1-1                                                            --                        [1, 577, 1024]            --                             --                   --                        --                        False
│    └─LoraModel: 2-1                                                       --                        --                        --                             --                   --                        --                        False
│    │    └─CLIPModel: 3-1                                                  --                        [1, 577, 1024]            (435,022,081)             100.00%                   --                        2,511,511,808             False
==========================================================================================================================================================================================================================================================
Total params: 435,022,081
Trainable params: 0
Non-trainable params: 435,022,081
Total mult-adds (Units.GIGABYTES): 2.51
==========================================================================================================================================================================================================================================================
Input size (MB): 1.35
Forward/backward pass size (MB): 2479.10
Params size (MB): 1740.08
Estimated Total Size (MB): 4220.54
==========================================================================================================================================================================================================================================================
```

### Model Structure
```txt
ClipBaselineModel(
  (model): PeftModel(
    (base_model): LoraModel(
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
                  (k_proj): lora.Linear(
                    (base_layer): Linear(in_features=1024, out_features=1024, bias=True)
                    (lora_dropout): ModuleDict(
                      (default): Dropout(p=0.1, inplace=False)
                    )
                    (lora_A): ModuleDict(
                      (default): Linear(in_features=1024, out_features=16, bias=False)
                    )
                    (lora_B): ModuleDict(
                      (default): Linear(in_features=16, out_features=1024, bias=False)
                    )
                    (lora_embedding_A): ParameterDict()
                    (lora_embedding_B): ParameterDict()
                    (lora_magnitude_vector): ModuleDict()
                  )
                  (v_proj): lora.Linear(
                    (base_layer): Linear(in_features=1024, out_features=1024, bias=True)
                    (lora_dropout): ModuleDict(
                      (default): Dropout(p=0.1, inplace=False)
                    )
                    (lora_A): ModuleDict(
                      (default): Linear(in_features=1024, out_features=16, bias=False)
                    )
                    (lora_B): ModuleDict(
                      (default): Linear(in_features=16, out_features=1024, bias=False)
                    )
                    (lora_embedding_A): ParameterDict()
                    (lora_embedding_B): ParameterDict()
                    (lora_magnitude_vector): ModuleDict()
                  )
                  (q_proj): lora.Linear(
                    (base_layer): Linear(in_features=1024, out_features=1024, bias=True)
                    (lora_dropout): ModuleDict(
                      (default): Dropout(p=0.1, inplace=False)
                    )
                    (lora_A): ModuleDict(
                      (default): Linear(in_features=1024, out_features=16, bias=False)
                    )
                    (lora_B): ModuleDict(
                      (default): Linear(in_features=16, out_features=1024, bias=False)
                    )
                    (lora_embedding_A): ParameterDict()
                    (lora_embedding_B): ParameterDict()
                    (lora_magnitude_vector): ModuleDict()
                  )
                  (out_proj): lora.Linear(
                    (base_layer): Linear(in_features=1024, out_features=1024, bias=True)
                    (lora_dropout): ModuleDict(
                      (default): Dropout(p=0.1, inplace=False)
                    )
                    (lora_A): ModuleDict(
                      (default): Linear(in_features=1024, out_features=16, bias=False)
                    )
                    (lora_B): ModuleDict(
                      (default): Linear(in_features=16, out_features=1024, bias=False)
                    )
                    (lora_embedding_A): ParameterDict()
                    (lora_embedding_B): ParameterDict()
                    (lora_magnitude_vector): ModuleDict()
                  )
                )
                (layer_norm1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
                (mlp): CLIPMLP(
                  (activation_fn): QuickGELUActivation()
                  (fc1): lora.Linear(
                    (base_layer): Linear(in_features=1024, out_features=4096, bias=True)
                    (lora_dropout): ModuleDict(
                      (default): Dropout(p=0.1, inplace=False)
                    )
                    (lora_A): ModuleDict(
                      (default): Linear(in_features=1024, out_features=16, bias=False)
                    )
                    (lora_B): ModuleDict(
                      (default): Linear(in_features=16, out_features=4096, bias=False)
                    )
                    (lora_embedding_A): ParameterDict()
                    (lora_embedding_B): ParameterDict()
                    (lora_magnitude_vector): ModuleDict()
                  )
                  (fc2): lora.Linear(
                    (base_layer): Linear(in_features=4096, out_features=1024, bias=True)
                    (lora_dropout): ModuleDict(
                      (default): Dropout(p=0.1, inplace=False)
                    )
                    (lora_A): ModuleDict(
                      (default): Linear(in_features=4096, out_features=16, bias=False)
                    )
                    (lora_B): ModuleDict(
                      (default): Linear(in_features=16, out_features=1024, bias=False)
                    )
                    (lora_embedding_A): ParameterDict()
                    (lora_embedding_B): ParameterDict()
                    (lora_magnitude_vector): ModuleDict()
                  )
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
    )
  )
)
```

## LoRA Clip (torch.nn.Linear, torch.nn.Embedding)

### Flop

```json
{
    "model_type": "<class 'model.ClipBaselineModel.ClipBaselineModel'>",
    "flops": "Skipped",
    "params": "Skipped",
    "run_time": 0.06046105575561524,
    "test_result": [
        {
            "val_acc": 0.9773809313774109,
            "val_loss": 0.08776428550481796
        }
    ]
}
```

### Module summary

```txt
==========================================================================================================================================================================================================================================================
Layer (type:depth-idx)                                                      Input Shape               Output Shape              Param #                   Param %                   Kernel Shape              Mult-Adds                 Trainable
==========================================================================================================================================================================================================================================================
ClipBaselineModel                                                           [1, 3, 336, 336]          [1, 577, 1024]            --                             --                   --                        --                        False
├─PeftModel: 1-1                                                            --                        [1, 577, 1024]            --                             --                   --                        --                        False
│    └─LoraModel: 2-1                                                       --                        --                        --                             --                   --                        --                        False
│    │    └─CLIPModel: 3-1                                                  --                        [1, 577, 1024]            (438,571,489)             100.00%                   --                        2,551,722,240             False
==========================================================================================================================================================================================================================================================
Total params: 438,571,489
Trainable params: 0
Non-trainable params: 438,571,489
Total mult-adds (Units.GIGABYTES): 2.55
==========================================================================================================================================================================================================================================================
Input size (MB): 1.35
Forward/backward pass size (MB): 2630.58
Params size (MB): 1750.91
Estimated Total Size (MB): 4382.85
==========================================================================================================================================================================================================================================================
```

### Model Structure
```txt
ClipBaselineModel(
  (model): PeftModel(
    (base_model): LoraModel(
      (model): CLIPModel(
        (text_model): CLIPTextTransformer(
          (embeddings): CLIPTextEmbeddings(
            (token_embedding): lora.Embedding(
              (base_layer): Embedding(49408, 768)
              (lora_dropout): ModuleDict(
                (default): Dropout(p=0.1, inplace=False)
              )
              (lora_A): ModuleDict()
              (lora_B): ModuleDict()
              (lora_embedding_A): ParameterDict(  (default): Parameter containing: [torch.cuda.FloatTensor of size 16x49408 (cuda:0)])
              (lora_embedding_B): ParameterDict(  (default): Parameter containing: [torch.cuda.FloatTensor of size 768x16 (cuda:0)])
              (lora_magnitude_vector): ModuleDict()
            )
            (position_embedding): lora.Embedding(
              (base_layer): Embedding(77, 768)
              (lora_dropout): ModuleDict(
                (default): Dropout(p=0.1, inplace=False)
              )
              (lora_A): ModuleDict()
              (lora_B): ModuleDict()
              (lora_embedding_A): ParameterDict(  (default): Parameter containing: [torch.cuda.FloatTensor of size 16x77 (cuda:0)])
              (lora_embedding_B): ParameterDict(  (default): Parameter containing: [torch.cuda.FloatTensor of size 768x16 (cuda:0)])
              (lora_magnitude_vector): ModuleDict()
            )
          )
          (encoder): CLIPEncoder(
            (layers): ModuleList(
              (0-11): 12 x CLIPEncoderLayer(
                (self_attn): CLIPSdpaAttention(
                  (k_proj): lora.Linear(
                    (base_layer): Linear(in_features=768, out_features=768, bias=True)
                    (lora_dropout): ModuleDict(
                      (default): Dropout(p=0.1, inplace=False)
                    )
                    (lora_A): ModuleDict(
                      (default): Linear(in_features=768, out_features=16, bias=False)
                    )
                    (lora_B): ModuleDict(
                      (default): Linear(in_features=16, out_features=768, bias=False)
                    )
                    (lora_embedding_A): ParameterDict()
                    (lora_embedding_B): ParameterDict()
                    (lora_magnitude_vector): ModuleDict()
                  )
                  (v_proj): lora.Linear(
                    (base_layer): Linear(in_features=768, out_features=768, bias=True)
                    (lora_dropout): ModuleDict(
                      (default): Dropout(p=0.1, inplace=False)
                    )
                    (lora_A): ModuleDict(
                      (default): Linear(in_features=768, out_features=16, bias=False)
                    )
                    (lora_B): ModuleDict(
                      (default): Linear(in_features=16, out_features=768, bias=False)
                    )
                    (lora_embedding_A): ParameterDict()
                    (lora_embedding_B): ParameterDict()
                    (lora_magnitude_vector): ModuleDict()
                  )
                  (q_proj): lora.Linear(
                    (base_layer): Linear(in_features=768, out_features=768, bias=True)
                    (lora_dropout): ModuleDict(
                      (default): Dropout(p=0.1, inplace=False)
                    )
                    (lora_A): ModuleDict(
                      (default): Linear(in_features=768, out_features=16, bias=False)
                    )
                    (lora_B): ModuleDict(
                      (default): Linear(in_features=16, out_features=768, bias=False)
                    )
                    (lora_embedding_A): ParameterDict()
                    (lora_embedding_B): ParameterDict()
                    (lora_magnitude_vector): ModuleDict()
                  )
                  (out_proj): lora.Linear(
                    (base_layer): Linear(in_features=768, out_features=768, bias=True)
                    (lora_dropout): ModuleDict(
                      (default): Dropout(p=0.1, inplace=False)
                    )
                    (lora_A): ModuleDict(
                      (default): Linear(in_features=768, out_features=16, bias=False)
                    )
                    (lora_B): ModuleDict(
                      (default): Linear(in_features=16, out_features=768, bias=False)
                    )
                    (lora_embedding_A): ParameterDict()
                    (lora_embedding_B): ParameterDict()
                    (lora_magnitude_vector): ModuleDict()
                  )
                )
                (layer_norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (mlp): CLIPMLP(
                  (activation_fn): QuickGELUActivation()
                  (fc1): lora.Linear(
                    (base_layer): Linear(in_features=768, out_features=3072, bias=True)
                    (lora_dropout): ModuleDict(
                      (default): Dropout(p=0.1, inplace=False)
                    )
                    (lora_A): ModuleDict(
                      (default): Linear(in_features=768, out_features=16, bias=False)
                    )
                    (lora_B): ModuleDict(
                      (default): Linear(in_features=16, out_features=3072, bias=False)
                    )
                    (lora_embedding_A): ParameterDict()
                    (lora_embedding_B): ParameterDict()
                    (lora_magnitude_vector): ModuleDict()
                  )
                  (fc2): lora.Linear(
                    (base_layer): Linear(in_features=3072, out_features=768, bias=True)
                    (lora_dropout): ModuleDict(
                      (default): Dropout(p=0.1, inplace=False)
                    )
                    (lora_A): ModuleDict(
                      (default): Linear(in_features=3072, out_features=16, bias=False)
                    )
                    (lora_B): ModuleDict(
                      (default): Linear(in_features=16, out_features=768, bias=False)
                    )
                    (lora_embedding_A): ParameterDict()
                    (lora_embedding_B): ParameterDict()
                    (lora_magnitude_vector): ModuleDict()
                  )
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
            (position_embedding): lora.Embedding(
              (base_layer): Embedding(577, 1024)
              (lora_dropout): ModuleDict(
                (default): Dropout(p=0.1, inplace=False)
              )
              (lora_A): ModuleDict()
              (lora_B): ModuleDict()
              (lora_embedding_A): ParameterDict(  (default): Parameter containing: [torch.cuda.FloatTensor of size 16x577 (cuda:0)])
              (lora_embedding_B): ParameterDict(  (default): Parameter containing: [torch.cuda.FloatTensor of size 1024x16 (cuda:0)])
              (lora_magnitude_vector): ModuleDict()
            )
          )
          (pre_layrnorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
          (encoder): CLIPEncoder(
            (layers): ModuleList(
              (0-23): 24 x CLIPEncoderLayer(
                (self_attn): CLIPSdpaAttention(
                  (k_proj): lora.Linear(
                    (base_layer): Linear(in_features=1024, out_features=1024, bias=True)
                    (lora_dropout): ModuleDict(
                      (default): Dropout(p=0.1, inplace=False)
                    )
                    (lora_A): ModuleDict(
                      (default): Linear(in_features=1024, out_features=16, bias=False)
                    )
                    (lora_B): ModuleDict(
                      (default): Linear(in_features=16, out_features=1024, bias=False)
                    )
                    (lora_embedding_A): ParameterDict()
                    (lora_embedding_B): ParameterDict()
                    (lora_magnitude_vector): ModuleDict()
                  )
                  (v_proj): lora.Linear(
                    (base_layer): Linear(in_features=1024, out_features=1024, bias=True)
                    (lora_dropout): ModuleDict(
                      (default): Dropout(p=0.1, inplace=False)
                    )
                    (lora_A): ModuleDict(
                      (default): Linear(in_features=1024, out_features=16, bias=False)
                    )
                    (lora_B): ModuleDict(
                      (default): Linear(in_features=16, out_features=1024, bias=False)
                    )
                    (lora_embedding_A): ParameterDict()
                    (lora_embedding_B): ParameterDict()
                    (lora_magnitude_vector): ModuleDict()
                  )
                  (q_proj): lora.Linear(
                    (base_layer): Linear(in_features=1024, out_features=1024, bias=True)
                    (lora_dropout): ModuleDict(
                      (default): Dropout(p=0.1, inplace=False)
                    )
                    (lora_A): ModuleDict(
                      (default): Linear(in_features=1024, out_features=16, bias=False)
                    )
                    (lora_B): ModuleDict(
                      (default): Linear(in_features=16, out_features=1024, bias=False)
                    )
                    (lora_embedding_A): ParameterDict()
                    (lora_embedding_B): ParameterDict()
                    (lora_magnitude_vector): ModuleDict()
                  )
                  (out_proj): lora.Linear(
                    (base_layer): Linear(in_features=1024, out_features=1024, bias=True)
                    (lora_dropout): ModuleDict(
                      (default): Dropout(p=0.1, inplace=False)
                    )
                    (lora_A): ModuleDict(
                      (default): Linear(in_features=1024, out_features=16, bias=False)
                    )
                    (lora_B): ModuleDict(
                      (default): Linear(in_features=16, out_features=1024, bias=False)
                    )
                    (lora_embedding_A): ParameterDict()
                    (lora_embedding_B): ParameterDict()
                    (lora_magnitude_vector): ModuleDict()
                  )
                )
                (layer_norm1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
                (mlp): CLIPMLP(
                  (activation_fn): QuickGELUActivation()
                  (fc1): lora.Linear(
                    (base_layer): Linear(in_features=1024, out_features=4096, bias=True)
                    (lora_dropout): ModuleDict(
                      (default): Dropout(p=0.1, inplace=False)
                    )
                    (lora_A): ModuleDict(
                      (default): Linear(in_features=1024, out_features=16, bias=False)
                    )
                    (lora_B): ModuleDict(
                      (default): Linear(in_features=16, out_features=4096, bias=False)
                    )
                    (lora_embedding_A): ParameterDict()
                    (lora_embedding_B): ParameterDict()
                    (lora_magnitude_vector): ModuleDict()
                  )
                  (fc2): lora.Linear(
                    (base_layer): Linear(in_features=4096, out_features=1024, bias=True)
                    (lora_dropout): ModuleDict(
                      (default): Dropout(p=0.1, inplace=False)
                    )
                    (lora_A): ModuleDict(
                      (default): Linear(in_features=4096, out_features=16, bias=False)
                    )
                    (lora_B): ModuleDict(
                      (default): Linear(in_features=16, out_features=1024, bias=False)
                    )
                    (lora_embedding_A): ParameterDict()
                    (lora_embedding_B): ParameterDict()
                    (lora_magnitude_vector): ModuleDict()
                  )
                )
                (layer_norm2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              )
            )
          )
          (post_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        )
        (visual_projection): lora.Linear(
          (base_layer): Linear(in_features=1024, out_features=768, bias=False)
          (lora_dropout): ModuleDict(
            (default): Dropout(p=0.1, inplace=False)
          )
          (lora_A): ModuleDict(
            (default): Linear(in_features=1024, out_features=16, bias=False)
          )
          (lora_B): ModuleDict(
            (default): Linear(in_features=16, out_features=768, bias=False)
          )
          (lora_embedding_A): ParameterDict()
          (lora_embedding_B): ParameterDict()
          (lora_magnitude_vector): ModuleDict()
        )
        (text_projection): lora.Linear(
          (base_layer): Linear(in_features=768, out_features=768, bias=False)
          (lora_dropout): ModuleDict(
            (default): Dropout(p=0.1, inplace=False)
          )
          (lora_A): ModuleDict(
            (default): Linear(in_features=768, out_features=16, bias=False)
          )
          (lora_B): ModuleDict(
            (default): Linear(in_features=16, out_features=768, bias=False)
          )
          (lora_embedding_A): ParameterDict()
          (lora_embedding_B): ParameterDict()
          (lora_magnitude_vector): ModuleDict()
        )
      )
    )
  )
)
```

## LoRA Clip (torch.nn.Linear Text Encoder, torch.nn.Embedding Text Encoder)

### Flop

```json
{
    "model_type": "<class 'model.ClipBaselineModel.ClipBaselineModel'>",
    "flops": "Skipped",
    "params": "Skipped",
    "run_time": 0.03890585708618164,
    "test_result": [
        {
            "val_acc": 0.954365074634552,
            "val_loss": 0.15729843080043793
        }
    ]
}
```

### Module summary

```txt
==========================================================================================================================================================================================================================================================
Layer (type:depth-idx)                                                      Input Shape               Output Shape              Param #                   Param %                   Kernel Shape              Mult-Adds                 Trainable
==========================================================================================================================================================================================================================================================
ClipBaselineModel                                                           [1, 3, 336, 336]          [1, 577, 1024]            --                             --                   --                        --                        False
├─PeftModel: 1-1                                                            --                        [1, 577, 1024]            --                             --                   --                        --                        False
│    └─LoraModel: 2-1                                                       --                        --                        --                             --                   --                        --                        False
│    │    └─CLIPModel: 3-1                                                  --                        [1, 577, 1024]            (431,467,985)             100.00%                   --                        2,544,644,352             False
==========================================================================================================================================================================================================================================================
Total params: 431,467,985
Trainable params: 0
Non-trainable params: 431,467,985
Total mult-adds (Units.GIGABYTES): 2.54
==========================================================================================================================================================================================================================================================
Input size (MB): 1.35
Forward/backward pass size (MB): 1598.96
Params size (MB): 1722.60
Estimated Total Size (MB): 3322.91
==========================================================================================================================================================================================================================================================
```

### Model Structure

```txt
ClipBaselineModel(
  (model): PeftModel(
    (base_model): LoraModel(
      (model): CLIPModel(
        (text_model): CLIPTextTransformer(
          (embeddings): CLIPTextEmbeddings(
            (token_embedding): lora.Embedding(
              (base_layer): Embedding(49408, 768)
              (lora_dropout): ModuleDict(
                (default): Dropout(p=0.1, inplace=False)
              )
              (lora_A): ModuleDict()
              (lora_B): ModuleDict()
              (lora_embedding_A): ParameterDict(  (default): Parameter containing: [torch.cuda.FloatTensor of size 16x49408 (cuda:0)])
              (lora_embedding_B): ParameterDict(  (default): Parameter containing: [torch.cuda.FloatTensor of size 768x16 (cuda:0)])
              (lora_magnitude_vector): ModuleDict()
            )
            (position_embedding): lora.Embedding(
              (base_layer): Embedding(77, 768)
              (lora_dropout): ModuleDict(
                (default): Dropout(p=0.1, inplace=False)
              )
              (lora_A): ModuleDict()
              (lora_B): ModuleDict()
              (lora_embedding_A): ParameterDict(  (default): Parameter containing: [torch.cuda.FloatTensor of size 16x77 (cuda:0)])
              (lora_embedding_B): ParameterDict(  (default): Parameter containing: [torch.cuda.FloatTensor of size 768x16 (cuda:0)])
              (lora_magnitude_vector): ModuleDict()
            )
          )
          (encoder): CLIPEncoder(
            (layers): ModuleList(
              (0-11): 12 x CLIPEncoderLayer(
                (self_attn): CLIPSdpaAttention(
                  (k_proj): lora.Linear(
                    (base_layer): Linear(in_features=768, out_features=768, bias=True)
                    (lora_dropout): ModuleDict(
                      (default): Dropout(p=0.1, inplace=False)
                    )
                    (lora_A): ModuleDict(
                      (default): Linear(in_features=768, out_features=16, bias=False)
                    )
                    (lora_B): ModuleDict(
                      (default): Linear(in_features=16, out_features=768, bias=False)
                    )
                    (lora_embedding_A): ParameterDict()
                    (lora_embedding_B): ParameterDict()
                    (lora_magnitude_vector): ModuleDict()
                  )
                  (v_proj): lora.Linear(
                    (base_layer): Linear(in_features=768, out_features=768, bias=True)
                    (lora_dropout): ModuleDict(
                      (default): Dropout(p=0.1, inplace=False)
                    )
                    (lora_A): ModuleDict(
                      (default): Linear(in_features=768, out_features=16, bias=False)
                    )
                    (lora_B): ModuleDict(
                      (default): Linear(in_features=16, out_features=768, bias=False)
                    )
                    (lora_embedding_A): ParameterDict()
                    (lora_embedding_B): ParameterDict()
                    (lora_magnitude_vector): ModuleDict()
                  )
                  (q_proj): lora.Linear(
                    (base_layer): Linear(in_features=768, out_features=768, bias=True)
                    (lora_dropout): ModuleDict(
                      (default): Dropout(p=0.1, inplace=False)
                    )
                    (lora_A): ModuleDict(
                      (default): Linear(in_features=768, out_features=16, bias=False)
                    )
                    (lora_B): ModuleDict(
                      (default): Linear(in_features=16, out_features=768, bias=False)
                    )
                    (lora_embedding_A): ParameterDict()
                    (lora_embedding_B): ParameterDict()
                    (lora_magnitude_vector): ModuleDict()
                  )
                  (out_proj): lora.Linear(
                    (base_layer): Linear(in_features=768, out_features=768, bias=True)
                    (lora_dropout): ModuleDict(
                      (default): Dropout(p=0.1, inplace=False)
                    )
                    (lora_A): ModuleDict(
                      (default): Linear(in_features=768, out_features=16, bias=False)
                    )
                    (lora_B): ModuleDict(
                      (default): Linear(in_features=16, out_features=768, bias=False)
                    )
                    (lora_embedding_A): ParameterDict()
                    (lora_embedding_B): ParameterDict()
                    (lora_magnitude_vector): ModuleDict()
                  )
                )
                (layer_norm1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (mlp): CLIPMLP(
                  (activation_fn): QuickGELUActivation()
                  (fc1): lora.Linear(
                    (base_layer): Linear(in_features=768, out_features=3072, bias=True)
                    (lora_dropout): ModuleDict(
                      (default): Dropout(p=0.1, inplace=False)
                    )
                    (lora_A): ModuleDict(
                      (default): Linear(in_features=768, out_features=16, bias=False)
                    )
                    (lora_B): ModuleDict(
                      (default): Linear(in_features=16, out_features=3072, bias=False)
                    )
                    (lora_embedding_A): ParameterDict()
                    (lora_embedding_B): ParameterDict()
                    (lora_magnitude_vector): ModuleDict()
                  )
                  (fc2): lora.Linear(
                    (base_layer): Linear(in_features=3072, out_features=768, bias=True)
                    (lora_dropout): ModuleDict(
                      (default): Dropout(p=0.1, inplace=False)
                    )
                    (lora_A): ModuleDict(
                      (default): Linear(in_features=3072, out_features=16, bias=False)
                    )
                    (lora_B): ModuleDict(
                      (default): Linear(in_features=16, out_features=768, bias=False)
                    )
                    (lora_embedding_A): ParameterDict()
                    (lora_embedding_B): ParameterDict()
                    (lora_magnitude_vector): ModuleDict()
                  )
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
        (visual_projection): lora.Linear(
          (base_layer): Linear(in_features=1024, out_features=768, bias=False)
          (lora_dropout): ModuleDict(
            (default): Dropout(p=0.1, inplace=False)
          )
          (lora_A): ModuleDict(
            (default): Linear(in_features=1024, out_features=16, bias=False)
          )
          (lora_B): ModuleDict(
            (default): Linear(in_features=16, out_features=768, bias=False)
          )
          (lora_embedding_A): ParameterDict()
          (lora_embedding_B): ParameterDict()
          (lora_magnitude_vector): ModuleDict()
        )
        (text_projection): lora.Linear(
          (base_layer): Linear(in_features=768, out_features=768, bias=False)
          (lora_dropout): ModuleDict(
            (default): Dropout(p=0.1, inplace=False)
          )
          (lora_A): ModuleDict(
            (default): Linear(in_features=768, out_features=16, bias=False)
          )
          (lora_B): ModuleDict(
            (default): Linear(in_features=16, out_features=768, bias=False)
          )
          (lora_embedding_A): ParameterDict()
          (lora_embedding_B): ParameterDict()
          (lora_magnitude_vector): ModuleDict()
        )
      )
    )
  )
)
```


## LoRA Clip (torch.nn.Linear Vision Encoder, torch.nn.Embedding Vision Encoder)

### Flop

```json
{
    "model_type": "<class 'model.ClipBaselineModel.ClipBaselineModel'>",
    "flops": "Skipped",
    "params": "Skipped",
    "run_time": 0.05066342544555664,
    "test_result": [
        {
            "val_acc": 0.9690476059913635,
            "val_loss": 0.11764460057020187
        }
    ]
}
```

### Module summary

```txt
==========================================================================================================================================================================================================================================================
Layer (type:depth-idx)                                                      Input Shape               Output Shape              Param #                   Param %                   Kernel Shape              Mult-Adds                 Trainable
==========================================================================================================================================================================================================================================================
ClipBaselineModel                                                           [1, 3, 336, 336]          [1, 577, 1024]            --                             --                   --                        --                        False
├─PeftModel: 1-1                                                            --                        [1, 577, 1024]            --                             --                   --                        --                        False
│    └─LoraModel: 2-1                                                       --                        --                        --                             --                   --                        --                        False
│    │    └─CLIPModel: 3-1                                                  --                        [1, 577, 1024]            (435,100,945)             100.00%                   --                        2,511,909,120             False
==========================================================================================================================================================================================================================================================
Total params: 435,100,945
Trainable params: 0
Non-trainable params: 435,100,945
Total mult-adds (Units.GIGABYTES): 2.51
==========================================================================================================================================================================================================================================================
Input size (MB): 1.35
Forward/backward pass size (MB): 2479.20
Params size (MB): 1740.30
Estimated Total Size (MB): 4220.86
==========================================================================================================================================================================================================================================================
```

### Model Structure

```txt
ClipBaselineModel(
  (model): PeftModel(
    (base_model): LoraModel(
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
            (position_embedding): lora.Embedding(
              (base_layer): Embedding(577, 1024)
              (lora_dropout): ModuleDict(
                (default): Dropout(p=0.1, inplace=False)
              )
              (lora_A): ModuleDict()
              (lora_B): ModuleDict()
              (lora_embedding_A): ParameterDict(  (default): Parameter containing: [torch.cuda.FloatTensor of size 16x577 (cuda:0)])
              (lora_embedding_B): ParameterDict(  (default): Parameter containing: [torch.cuda.FloatTensor of size 1024x16 (cuda:0)])
              (lora_magnitude_vector): ModuleDict()
            )
          )
          (pre_layrnorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
          (encoder): CLIPEncoder(
            (layers): ModuleList(
              (0-23): 24 x CLIPEncoderLayer(
                (self_attn): CLIPSdpaAttention(
                  (k_proj): lora.Linear(
                    (base_layer): Linear(in_features=1024, out_features=1024, bias=True)
                    (lora_dropout): ModuleDict(
                      (default): Dropout(p=0.1, inplace=False)
                    )
                    (lora_A): ModuleDict(
                      (default): Linear(in_features=1024, out_features=16, bias=False)
                    )
                    (lora_B): ModuleDict(
                      (default): Linear(in_features=16, out_features=1024, bias=False)
                    )
                    (lora_embedding_A): ParameterDict()
                    (lora_embedding_B): ParameterDict()
                    (lora_magnitude_vector): ModuleDict()
                  )
                  (v_proj): lora.Linear(
                    (base_layer): Linear(in_features=1024, out_features=1024, bias=True)
                    (lora_dropout): ModuleDict(
                      (default): Dropout(p=0.1, inplace=False)
                    )
                    (lora_A): ModuleDict(
                      (default): Linear(in_features=1024, out_features=16, bias=False)
                    )
                    (lora_B): ModuleDict(
                      (default): Linear(in_features=16, out_features=1024, bias=False)
                    )
                    (lora_embedding_A): ParameterDict()
                    (lora_embedding_B): ParameterDict()
                    (lora_magnitude_vector): ModuleDict()
                  )
                  (q_proj): lora.Linear(
                    (base_layer): Linear(in_features=1024, out_features=1024, bias=True)
                    (lora_dropout): ModuleDict(
                      (default): Dropout(p=0.1, inplace=False)
                    )
                    (lora_A): ModuleDict(
                      (default): Linear(in_features=1024, out_features=16, bias=False)
                    )
                    (lora_B): ModuleDict(
                      (default): Linear(in_features=16, out_features=1024, bias=False)
                    )
                    (lora_embedding_A): ParameterDict()
                    (lora_embedding_B): ParameterDict()
                    (lora_magnitude_vector): ModuleDict()
                  )
                  (out_proj): lora.Linear(
                    (base_layer): Linear(in_features=1024, out_features=1024, bias=True)
                    (lora_dropout): ModuleDict(
                      (default): Dropout(p=0.1, inplace=False)
                    )
                    (lora_A): ModuleDict(
                      (default): Linear(in_features=1024, out_features=16, bias=False)
                    )
                    (lora_B): ModuleDict(
                      (default): Linear(in_features=16, out_features=1024, bias=False)
                    )
                    (lora_embedding_A): ParameterDict()
                    (lora_embedding_B): ParameterDict()
                    (lora_magnitude_vector): ModuleDict()
                  )
                )
                (layer_norm1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
                (mlp): CLIPMLP(
                  (activation_fn): QuickGELUActivation()
                  (fc1): lora.Linear(
                    (base_layer): Linear(in_features=1024, out_features=4096, bias=True)
                    (lora_dropout): ModuleDict(
                      (default): Dropout(p=0.1, inplace=False)
                    )
                    (lora_A): ModuleDict(
                      (default): Linear(in_features=1024, out_features=16, bias=False)
                    )
                    (lora_B): ModuleDict(
                      (default): Linear(in_features=16, out_features=4096, bias=False)
                    )
                    (lora_embedding_A): ParameterDict()
                    (lora_embedding_B): ParameterDict()
                    (lora_magnitude_vector): ModuleDict()
                  )
                  (fc2): lora.Linear(
                    (base_layer): Linear(in_features=4096, out_features=1024, bias=True)
                    (lora_dropout): ModuleDict(
                      (default): Dropout(p=0.1, inplace=False)
                    )
                    (lora_A): ModuleDict(
                      (default): Linear(in_features=4096, out_features=16, bias=False)
                    )
                    (lora_B): ModuleDict(
                      (default): Linear(in_features=16, out_features=1024, bias=False)
                    )
                    (lora_embedding_A): ParameterDict()
                    (lora_embedding_B): ParameterDict()
                    (lora_magnitude_vector): ModuleDict()
                  )
                )
                (layer_norm2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              )
            )
          )
          (post_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        )
        (visual_projection): lora.Linear(
          (base_layer): Linear(in_features=1024, out_features=768, bias=False)
          (lora_dropout): ModuleDict(
            (default): Dropout(p=0.1, inplace=False)
          )
          (lora_A): ModuleDict(
            (default): Linear(in_features=1024, out_features=16, bias=False)
          )
          (lora_B): ModuleDict(
            (default): Linear(in_features=16, out_features=768, bias=False)
          )
          (lora_embedding_A): ParameterDict()
          (lora_embedding_B): ParameterDict()
          (lora_magnitude_vector): ModuleDict()
        )
        (text_projection): lora.Linear(
          (base_layer): Linear(in_features=768, out_features=768, bias=False)
          (lora_dropout): ModuleDict(
            (default): Dropout(p=0.1, inplace=False)
          )
          (lora_A): ModuleDict(
            (default): Linear(in_features=768, out_features=16, bias=False)
          )
          (lora_B): ModuleDict(
            (default): Linear(in_features=16, out_features=768, bias=False)
          )
          (lora_embedding_A): ParameterDict()
          (lora_embedding_B): ParameterDict()
          (lora_magnitude_vector): ModuleDict()
        )
      )
    )
  )
)
