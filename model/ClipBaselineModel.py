import lightning as L
import torch
import torch.nn.functional as F

from transformers import CLIPModel, CLIPProcessor
from transformers.models.clip.modeling_clip import clip_loss, CLIPOutput
from torch.optim.lr_scheduler import CosineAnnealingLR


class ClipBaselineModel(L.LightningModule):
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
        model_name: str = "openai/clip-vit-large-patch14-336",
        use_prompt: bool = True,
    ):
        super().__init__()

        self.model_name = model_name
        self.model = CLIPModel.from_pretrained(model_name)

        self.build_params(model_name=model_name, use_prompt=use_prompt)

        # just for forward
        self.model.requires_grad_(False)

        self.save_hyperparameters()
        return

    @torch.no_grad()
    @torch.inference_mode()
    def build_params(self, model_name: str, use_prompt: bool = True):
        processor = CLIPProcessor.from_pretrained(model_name)

        labels = (
            list(self.DEFAULT_MAPPING.values())
            if use_prompt
            else list(self.DEFAULT_MAPPING.keys())
        )

        inputs = processor(text=labels, return_tensors="pt", padding=True)

        self.input_ids = inputs["input_ids"]
        self.attention_mask = inputs["attention_mask"]

        self.input_ids.requires_grad_(False)
        self.attention_mask.requires_grad_(False)

        del processor, inputs

        return

    def forward(self, pixel_values: torch.Tensor):

        model_output: CLIPOutput = self.model(
            pixel_values=pixel_values,
            input_ids=self.input_ids.to(self.device),
            attention_mask=self.attention_mask.to(self.device),
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
        raise NotImplementedError(
            "ClipBaselineModel does not support training step. Use ClipLoRaHARModel instead."
        )

    @torch.inference_mode()
    def validation_step(self, batch, batch_idx):
        pixel_values, label_values = batch

        model_output: CLIPOutput = self(pixel_values=pixel_values)

        logits_per_image = model_output.logits_per_image

        probs = torch.softmax(logits_per_image, dim=1)
        preds = torch.argmax(probs, dim=1)

        acc = (preds == label_values).float().mean()
        loss = F.cross_entropy(logits_per_image, label_values)
        self.log_dict({"val_acc": acc, "val_loss": loss}, sync_dist=True)

        return

    @torch.inference_mode()
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        pixel_values = batch
        return self(pixel_values=pixel_values)
