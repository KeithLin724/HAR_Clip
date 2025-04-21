# 開發文件

## 簡介

本專案使用 `pytorch-lightning` 框架進行開發。`pytorch-lightning` 易於使用且結構化。以下示範各訓練組件的範例。

**若需新增或修改程式碼，請先建立新分支，然後透過 pull request 合併至 ****`main`**** 分支。**

## 資料集

我們已將資料集封裝為 PyTorch Lightning 相容的格式。主要類別為 `MotionDataset`（請參考 [data/MotionDataset.py](./data/MotionDataset.py)）。若要自動載入並預處理資料，請使用 `MotionDataModule`（請參考 [data/MotionDataModule.py](./data/MotionDataModule.py)）。如需自定義資料處理，請在 `data/` 資料夾新增檔案，並確保符合 PyTorch Lightning 規範。

### MotionDataModule 範例

```python
# 設定
DATASET_PATH = "./Human Action Recognition"
BATCH_SIZE = 64
VAL_SIZE = 0.2

# 載入資料描述
data_desc = MotionDataDescription.build_from_folder(DATASET_PATH)

# 建立 DataModule
datamodule = MotionDataModule(data_desc, batch_size=BATCH_SIZE, val_size=VAL_SIZE)
```

若需新增資料相關類別，請將其放在 `data/` 資料夾中。修改現有程式碼時，請**保留現有 API**或**新增 API**以支援其他功能（建議）。

## 模型

有關模型定義，請參考 PyTorch Lightning 官方文件：
[https://lightning.ai/docs/pytorch/stable/starter/introduction.html](https://lightning.ai/docs/pytorch/stable/starter/introduction.html)

### Lightning Module 範例

```python
import lightning as L

class Model(L.LightningModule):
    def __init__(self, ...):
        super().__init__()
        self.save_hyperparameters()

    def forward(self, ...):
        ...

    def configure_optimizers(self):
        optimizer = ...
        return optimizer
    
    def training_step(self, batch, batch_idx):
        loss = ...
        return loss

    def validation_step(self, batch, batch_idx):
        ...
    
    def predict_step(self, batch, batch_idx):
        ...
```

如需新增模型相關類別，請將檔案放在 `model/` 資料夾。修改現有模型時，請**保留現有 API**或**新增 API**（強烈建議）。

## 訓練器

`Trainer` 類別負責訓練流程，並支援記錄與回調功能。

### Trainer 範例

```python
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

LOG_PATH = "logs"
LOG_NAME = "clip_har"

# 設定日誌記錄器
 tb_logger = TensorBoardLogger(LOG_PATH, name=LOG_NAME)

# 設定檢查點回調
 checkpoint_callback = ModelCheckpoint(
     monitor="val_loss",
     filename="model-{epoch:02d}-{val_loss:.2f}",
     save_top_k=3,
     mode="min",
     save_last=True,
 )

# 設定早期停止回調
 early_stop_callback = EarlyStopping(
     monitor="val_loss",
     patience=10,
     mode="min",
 )

# 初始化 Trainer
 trainer = L.Trainer(
     callbacks=[checkpoint_callback, early_stop_callback],
     logger=tb_logger,
     max_epochs=EPOCH,
     log_every_n_steps=20,
     # fast_dev_run=True,
 )

# 開始訓練
 trainer.fit(model, datamodule=datamodule)
```

若要快速驗證訓練流程是否正常，請設置 `fast_dev_run=True`。驗證後，將其設置為 `fast_dev_run=False` 或註解此行。

> 要即時監控訓練過程，請執行：
>
> ```bash
> tensorboard --logdir {LOG_PATH}/{LOG_NAME}
> ```
