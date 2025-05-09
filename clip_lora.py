# %%
from transformers import CLIPModel, CLIPProcessor
import peft
from peft import LoraConfig, get_peft_model, TaskType
import torch

# %%
model_name = "openai/clip-vit-large-patch14-336"

model = CLIPModel.from_pretrained(model_name)

print(model)

# %%
target_modules = [
    name
    for name, module in model.named_modules()
    if isinstance(module, (torch.nn.Linear, torch.nn.Embedding))
]

# for i, (name, module) in enumerate(model.named_modules()):
#     if isinstance(module, torch.nn.Linear):
#         # print(f"{i}: {name} -> {module}")
#         target_modules.append(name)

#     if isinstance(module, torch.nn.LayerNorm):
#         # print(f"{i}: {name} -> {module}")
#         target_modules.append(name)

#     if isinstance(module, torch.nn.Embedding):
#         target_modules.append(name)


# %%
peft_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=target_modules,
    bias="none",
)

model.eval()
# %%
peft_model = get_peft_model(model, peft_config)
peft_model.print_trainable_parameters()

# %%
print(peft_model)

# %%
