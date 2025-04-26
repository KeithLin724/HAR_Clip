import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from dadaptation import DAdaptAdam

class CNNHARModel(L.LightningModule):
    """
    基於 CNN 的 Human Activity Recognition 模型。
    輸入: pixel_values (B,3,H,W)
    輸出: logits (B, num_labels)
    """
    def __init__(
        self,
        labels: list[str],
        input_channels: int = 3,
        conv_filters: list[int] = [16, 32, 64],
        fc_hidden_dim: int = 128,
        dropout_rate_conv: float = 0.25,
        dropout_rate_fc: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.labels = labels
        num_classes = len(labels)

        # build conv blocks
        layers = []
        in_ch = input_channels
        for out_ch in conv_filters:
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2,2),
                nn.Dropout(dropout_rate_conv),
            ]
            in_ch = out_ch
        self.conv_layers = nn.Sequential(*layers)

        # adaptive pool → flatten
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1,1))
        # classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate_fc),
            nn.Linear(conv_filters[-1], fc_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate_fc),
            nn.Linear(fc_hidden_dim, num_classes)
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(pixel_values)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)          # (B, conv_filters[-1])
        logits = self.classifier(x)      # (B, num_classes)
        return logits

    def configure_optimizers(self):
        """
        配置優化器。

        Returns:
            Optimizer: 使用 DAdaptAdam 優化器。
        """
        optimizer = DAdaptAdam(self.parameters())
        return optimizer

    def training_step(self, batch, batch_idx):
        """
        訓練步驟。

        Args:
            batch: 包含輸入圖像和標籤的批次數據。
            batch_idx: 批次索引。

        Returns:
            torch.Tensor: 訓練損失。
        """
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log_dict({"train_loss": loss, "train_acc": acc})
        return loss

    @torch.inference_mode()
    def validation_step(self, batch, batch_idx):
        """
        驗證步驟。

        Args:
            batch: 包含輸入圖像和標籤的批次數據。
            batch_idx: 批次索引。
        """
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log_dict({"val_loss": loss, "val_acc": acc})
        return

    def predict_step(self, batch, batch_idx):
        """
        預測步驟。

        Args:
            batch: 輸入圖像批次數據。
            batch_idx: 批次索引。

        Returns:
            torch.Tensor: 模型預測結果。
        """
        return self(batch)