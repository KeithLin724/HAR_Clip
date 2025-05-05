import lightning as L
from transformers import CLIPModel, CLIPProcessor
import torch
import torch.nn as nn
from typing import Callable
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from dadaptation import DAdaptAdam
from .Cliploss import ClipLoss
from transformers import CLIPTokenizer, CLIPProcessor

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
    ):
        super().__init__()
        label_num = len(labels)

        # ---- 1. Load CLIP backbone ----
        self.model: CLIPModel = CLIPModel.from_pretrained(model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)

        # ---- 2. Freeze backbone (optional) ----
        self.model.vision_model.requires_grad_(False)
        self.model.text_model.requires_grad_(False)

        # ---- 3. Linear classifier head ----
        self.weight_image = nn.Linear(CLIP_FEATURES, label_num)

        # ---- 4. Text prompt features ----
        if prompt_mapping is None:
            prompt_mapping = self.DEFAULT_MAPPING

        self.class_names = list(prompt_mapping.keys())
        self.prompt_texts = list(prompt_mapping.values())

        text_tokens = self.tokenizer(
            self.prompt_texts, return_tensors='pt', padding=True, truncation=True
        )
        with torch.no_grad():
            text_features = self.model.get_text_features(**text_tokens)
            text_features = F.normalize(text_features, dim=-1)

        # ✅ register_buffer 讓 text_features 跟著 model 移動，不需更新梯度
        self.register_buffer("label_text_features", text_features)

        # ---- 5. Loss ----
        self.clip_loss_fn = ClipLoss()
        return

    def compute_features(self, x):
        device = self.model.device
        self.input_ids = self.input_ids.to(device)
        self.attention_mask = self.attention_mask.to(device)

        outputs = self.model(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            pixel_values=x,
            return_dict=True,
        )

        image_features = outputs.image_embeds       # shape: [B, D]
        text_features  = outputs.text_embeds        # shape: [num_labels, D]

        logit_scale = self.model.logit_scale  # 這樣保留可導圖

        # 再用 torch.exp 保留計算圖
        logits = torch.exp(logit_scale) * image_feat @ text_feat.T
        return image_features, text_features, logit_scale

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
        x, y = batch  # x: images, y: class indices

        # 1. Image features
        image_feat = self.model.get_image_features(pixel_values=x)  # [B, D]
        image_feat = F.normalize(image_feat, dim=-1)

        # 2. Text features
        text_feat = self.label_text_features.to(image_feat.device)  # [C, D]
        text_feat = F.normalize(text_feat, dim=-1)

        # ✅ 3. Logit scale（保留參數本身）
        logit_scale_raw = self.model.logit_scale  # 這是 nn.Parameter
        logit_scale = logit_scale_raw.exp()       # 再拿來用在 loss 裡

        # 4. 使用 ClipLoss 計算對比損失（需提供 targets）
        loss = self.clip_loss_fn(image_feat, text_feat, logit_scale, targets=y)

        # 5. 計算準確率
        logits = logit_scale * image_feat @ text_feat.T
        acc = (logits.argmax(dim=1) == y).float().mean()

        # 6. Logging
        self.log("train_loss", loss)
        self.log("train_acc", acc)

        return loss


    @torch.inference_mode()
    def validation_step(self, batch, batch_idx):
        x, y = batch

        # 1. Image features
        image_feat = self.model.get_image_features(pixel_values=x)
        image_feat = F.normalize(image_feat, dim=-1)

        # 2. Text features
        text_feat = self.label_text_features.to(image_feat.device)
        text_feat = F.normalize(text_feat, dim=-1)

        # ✅ 使用原始 logit_scale 參數
        logit_scale_raw = self.model.logit_scale
        logit_scale = logit_scale_raw.exp()

        # 3. Loss & accuracy
        loss = self.clip_loss_fn(image_feat, text_feat, logit_scale, targets=y)
        logits = logit_scale * image_feat @ text_feat.T
        acc = (logits.argmax(dim=1) == y).float().mean()

        self.log_dict({"val_loss": loss, "val_acc": acc})

    def predict_step(self, batch, batch_idx):
        return self(batch)
