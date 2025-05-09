import lightning as L
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, TaskType
from transformers import CLIPModel
from transformers import CLIPProcessor
from dataclasses import dataclass, asdict

from .ClipLoss import ClipLoss


@dataclass(slots=True, frozen=True)
class ClipLoRaConfig:
    model_name: str = "openai/clip-vit-large-patch14-336"
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    bias: str = "none"
    task_type: TaskType = TaskType.FEATURE_EXTRACTION
    target_modules: list[str] = None

    @staticmethod
    def build_target_modules(model: nn.Module):
        return [
            name
            for name, module in model.named_modules()
            if isinstance(module, (torch.nn.Linear, torch.nn.Embedding))
        ]


class ClipLoRaHARModel(L.LightningModule):
    def __init__(self, clip_config: dict):
        super().__init__()
        clip_config = ClipLoRaConfig(**clip_config)
        self.model = self.build_peft_model(clip_config)
        self.processor = CLIPProcessor.from_pretrained(clip_config.model_name)

        self.save_hyperparameters()
        return

    @staticmethod
    def build_peft_model(clip_config: ClipLoRaConfig):
        model = CLIPModel.from_pretrained(clip_config.model_name)
        model.eval()

        # Build target modules
        target_modules = (
            clip_config.target_modules
            if clip_config.target_modules is not None
            else ClipLoRaConfig.build_target_modules(self.model)
        )

        peft_config = LoraConfig(
            task_type=clip_config.task_type,
            inference_mode=False,
            r=clip_config.r,
            lora_alpha=clip_config.lora_alpha,
            lora_dropout=clip_config.lora_dropout,
            target_modules=target_modules,
            bias=clip_config.bias,
        )

        peft_model = get_peft_model(model, peft_config)
        peft_model.print_trainable_parameters()
        return peft_model

    @staticmethod
    def from_config(config: ClipLoRaConfig):
        return ClipLoRaHARModel(asdict(config))

    def print_trainable_parameters(self):
        self.model.print_trainable_parameters()
        return

    # about the forward

    def forward(self, x): ...

    def configure_optimizers(self): ...

    def training_step(self, batch, batch_idx): ...

    @torch.inference_mode()
    def validation_step(self, batch, batch_idx): ...

    @torch.inference_mode()
    def test_step(self, batch, batch_idx): ...

    @torch.inference_mode()
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)

    def inference_func(self):
        return lambda: None
