import lightning as L
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, TaskType
from transformers import CLIPModel, CLIPProcessor
from transformers.models.clip.modeling_clip import clip_loss, CLIPOutput
from dataclasses import dataclass, asdict
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Callable


@dataclass(slots=True)
class ClipLoRaConfig:
    model_name: str = "openai/clip-vit-large-patch14-336"
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    bias: str = "none"
    task_type: TaskType = None
    target_modules: list[str] = None

    @staticmethod
    def build_target_modules(
        model: nn.Module,
        condition: Callable = lambda name, module_x: isinstance(
            module_x, (torch.nn.Linear, torch.nn.Embedding)
        ),
    ):
        return [
            name for name, module in model.named_modules() if condition(name, module)
        ]

    def lazy_build_target_modules(
        self,
        condition: Callable = lambda name, module_x: isinstance(
            module_x, (torch.nn.Linear, torch.nn.Embedding)
        ),
    ):

        model = CLIPModel.from_pretrained(self.model_name)
        model.eval()

        # update the target_modules
        self.target_modules = self.build_target_modules(model, condition)

        return self


# TODO: Move processor in model


class ClipLoRaHARModel(L.LightningModule):
    DEFAULT_MAPPING = {
        "calling": "A photo of a person making a phone call",
        "clapping": "A photo of a person enthusiastically clapping their hands",
        "cycling": "A photo of a person riding a bicycle outdoors",
        "dancing": "A photo of a person dancing with expressive movement",
        "drinking": "A photo of a person drinking a beverage",
        "eating": "A photo of a person eating a meal",
        "fighting": "A photo of two people fighting or engaging in a physical altercation",
        "hugging": "A photo of two people hugging each other",
        "laughing": "A photo of a person laughing happily",
        "listening_to_music": "A photo of a person wearing headphones and listening to music",
        "running": "A photo of a person running at a steady pace",
        "sitting": "A photo of a person sitting on a chair or bench",
        "sleeping": "A photo of a person sleeping peacefully",
        "texting": "A photo of a person texting on a smartphone",
        "using_laptop": "A photo of a person using a laptop",
    }

    def __init__(self, clip_config: dict | ClipLoRaConfig):
        super().__init__()
        if isinstance(clip_config, dict):
            clip_config = ClipLoRaConfig(**clip_config)
        self.model_name = clip_config.model_name
        self.model = self.build_peft_model(clip_config)

        self.build_params(clip_config)

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
            else ClipLoRaConfig.build_target_modules(model)
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

    @torch.no_grad()
    @torch.inference_mode()
    def build_params(self, clip_config: ClipLoRaConfig):
        processor = CLIPProcessor.from_pretrained(clip_config.model_name)

        labels = list(self.DEFAULT_MAPPING.values())

        inputs = processor(text=labels, return_tensors="pt", padding=True)

        self.input_ids = inputs["input_ids"]
        self.attention_mask = inputs["attention_mask"]

        self.input_ids.requires_grad_(False)
        self.attention_mask.requires_grad_(False)

        del processor, inputs

        return

    def find_labels_tensor(
        self, query: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:

        self.input_ids = self.input_ids.to(self.device)
        self.attention_mask = self.attention_mask.to(self.device)

        input_ids = self.input_ids[query]
        attention_mask = self.attention_mask[query]
        return input_ids, attention_mask

    def forward(self, pixel_values: torch.Tensor, label_values: torch.Tensor):
        input_ids, attention_mask = self.find_labels_tensor(label_values)

        model_output: CLIPOutput = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
        )

        return model_output

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=0.01)
        total_steps = 100000

        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def training_step(self, batch, batch_idx):
        pixel_values, label_values = batch

        model_output: CLIPOutput = self(
            pixel_values=pixel_values, label_values=label_values
        )

        loss = clip_loss(model_output.logits_per_text)

        self.log_dict({"training_loss": loss})
        return loss

    @torch.inference_mode()
    def validation_step(self, batch, batch_idx):
        pixel_values, label_values = batch

        model_output: CLIPOutput = self(
            pixel_values=pixel_values, label_values=label_values
        )

        loss = clip_loss(model_output.logits_per_text)

        self.log_dict({"val_loss": loss}, sync_dist=True)
        return

    @torch.inference_mode()
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        pixel_values, label_values = batch
        return self(pixel_values=pixel_values, label_values=label_values)

    def inference_only_func(self):
        label_values = len(ClipLoRaHARModel.DEFAULT_MAPPING)

        def model_forward_only(pixel_values: torch.Tensor):
            model_output: CLIPOutput = self(
                pixel_values=pixel_values,
                label_values=torch.arange(label_values).to(pixel_values.device),
            )

            return model_output.logits_per_image

        return model_forward_only
