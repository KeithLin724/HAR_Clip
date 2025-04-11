import lightning as L
from transformers import CLIPModel, CLIPProcessor
import torch
import torch.nn as nn
from typing import Callable
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from dadaptation import DAdaptAdam

CLIP_FEATURES = 768


class ClipHARModel(L.LightningModule):
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

    def __init__(
        self,
        labels: list[str],
        prompt_mapping: dict = None,
        model_name: str = "openai/clip-vit-large-patch14-336",
        # dropout: float = 0.1,
    ):
        super().__init__()
        label_num = len(labels)

        self.model: CLIPModel = CLIPModel.from_pretrained(model_name)

        self.weight_image_image = nn.Linear(CLIP_FEATURES, CLIP_FEATURES, bias=False)
        self.weight_image = nn.Linear(CLIP_FEATURES, label_num)
        self.weight_image_learnable = nn.Linear(CLIP_FEATURES, label_num)

        # self.weight_image_text = nn.Linear(label_num, label_num)

        # self.dropout = nn.Dropout(dropout)

        if prompt_mapping is None:
            prompt_mapping = self.DEFAULT_MAPPING

        self.build_params(model_name, labels, prompt_mapping)

        self.model.requires_grad_(False)

        self.save_hyperparameters(ignore="prompt_format")
        return

    @torch.no_grad()
    def build_params(self, model_name: str, labels: list[str], prompt_mapping: dict):
        processor = CLIPProcessor.from_pretrained(model_name)

        labels = [prompt_mapping.get(label, label) for label in labels]

        inputs = processor(text=labels, return_tensors="pt", padding=True)

        self.input_ids = inputs["input_ids"]
        self.attention_mask = inputs["attention_mask"]

        self.input_ids.requires_grad_(False)
        self.attention_mask.requires_grad_(False)

        del processor, inputs

        return

    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        model_outputs = self.model.get_image_features(pixel_values=pixel_values)
        return self.weight_image_image(model_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = self.model.device

        self.input_ids = self.input_ids.to(device)
        self.attention_mask = self.attention_mask.to(device)

        model_outputs = self.model(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            pixel_values=x,
        )

        inputs_image_features = model_outputs.image_embeds

        logits_per_image = model_outputs.logits_per_image

        out = self.weight_image(self.weight_image_image(inputs_image_features))
        out += self.weight_image_learnable(inputs_image_features)
        # out += self.weight_image_text(logits_per_image)
        out += logits_per_image

        return out

    def configure_optimizers(self):
        # optimizer = optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-2)
        # total_steps = 100000

        # scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": scheduler,
        #         "interval": "step",
        #     },
        # }
        optimizer = DAdaptAdam(self.parameters())
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log_dict({"train_loss": loss, "train_acc": acc})

        return loss

    @torch.inference_mode()
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log_dict({"val_loss": loss, "val_acc": acc})

        return

    def predict_step(self, batch, batch_idx):
        return self(batch)
