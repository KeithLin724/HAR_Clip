import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from dadaptation import DAdaptAdam
import torch.optim as optim # 匯入 SGD 優化器
import torchvision.models as models # 匯入 torchvision 的模型庫

class CNNHARModel(L.LightningModule):
    """
    基於預訓練 ResNet 的 Human Activity Recognition 模型。
    輸入: pixel_values (B,3,H,W) - 假設輸入是標準的 3 通道影像
    輸出: logits (B, num_labels)
    """
    def __init__(
        self,
        labels: list[str],
        model_name: str = 'resnet18', # 可以選擇不同的預訓練模型, e.g., 'mobilenet_v2'
        fc_hidden_dim: int = 128,
        dropout_rate_fc: float = 0.6,
        learning_rate: float = 0.01 # SGD 需要指定學習率
    ):
        """
        Args:
            labels (list[str]): 類別標籤列表。
            model_name (str): 要使用的 torchvision 模型名稱 (e.g., 'resnet18', 'resnet34', 'mobilenet_v2')。
            fc_hidden_dim (int): 全連接層的隱藏層維度。
            dropout_rate_fc (float): 全連接層的 Dropout 率。
            learning_rate (float): SGD 優化器的學習率。
        """
        super().__init__()
        self.save_hyperparameters() # 儲存超參數, 包括 learning_rate
        self.labels = labels
        num_classes = len(labels)
        self.learning_rate = learning_rate # 將學習率存為屬性

        # 1. 載入預訓練模型
        # 使用最新的 weights API
        if model_name == 'resnet18':
            weights = models.ResNet18_Weights.DEFAULT
            backbone = models.resnet18(weights=weights)
            num_backbone_features = backbone.fc.in_features # 獲取 ResNet 最後一層 FC 的輸入特徵數
        elif model_name == 'resnet34':
            weights = models.ResNet34_Weights.DEFAULT
            backbone = models.resnet34(weights=weights)
            num_backbone_features = backbone.fc.in_features
        elif model_name == 'mobilenet_v2':
            weights = models.MobileNet_V2_Weights.DEFAULT
            backbone = models.mobilenet_v2(weights=weights)
            # MobileNetV2 的分類器是 backbone.classifier[1]
            num_backbone_features = backbone.classifier[1].in_features
        # 可以依此類推添加更多模型...
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")

        # 2. 移除原始模型的分類層
        if 'resnet' in model_name:
            # ResNet 系列: 移除最後的 fc 層
            self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        elif 'mobilenet' in model_name:
            # MobileNetV2: 移除最後的 classifier 層
            # MobileNetV2 的結構是 features -> adaptive_avg_pool -> classifier
            # 我們保留 features 和 adaptive_avg_pool
            self.feature_extractor = nn.Sequential(
                backbone.features,
                nn.AdaptiveAvgPool2d((1, 1)) # 確保輸出是 (B, C, 1, 1)
            )
        else:
             # 默認嘗試移除最後一層 (可能需要根據具體模型調整)
             self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])


        # 3. 建立新的分類器 (與原來的結構類似，但輸入維度不同)
        # 注意：輸入維度是預訓練模型提取的特徵維度
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate_fc),
            nn.Linear(num_backbone_features, fc_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate_fc),
            nn.Linear(fc_hidden_dim, num_classes)
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        前向傳播。
        輸入: pixel_values (B, 3, H, W)
        輸出: logits (B, num_classes)
        """
        # 提取特徵
        features = self.feature_extractor(pixel_values) # (B, num_backbone_features, 1, 1) 或 (B, num_backbone_features, H', W') 取決於模型結構
        # 展平特徵
        x = torch.flatten(features, 1) # (B, num_backbone_features)
        # 通過新的分類器
        logits = self.classifier(x)     # (B, num_classes)
        return logits

    def configure_optimizers(self):
        """
        配置優化器 - 改用 SGD。
        """
        # 可以選擇只優化新添加的分類器層，或者優化所有層（微調）
        # 這裡優化所有參數
        optimizer = optim.SGD(
            self.parameters(),
            lr=self.learning_rate, # 使用儲存的學習率
            momentum=0.9, # SGD 常用的動量值
            weight_decay=1e-4 # L2 正則化
        )
        # 可以選擇性地添加學習率調度器 (Learning Rate Scheduler)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        # return [optimizer], [scheduler]
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