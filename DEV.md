# Development Documentation

## Introduction

This project uses the `pytorch-lightning` framework for development. `pytorch-lightning` is user-friendly and modular. Below are examples for each training component.

**If you want to add or modify code, please create a new branch and submit a pull request to merge into the ****`main`**** branch.**

## Dataset

We have packaged the dataset in a PyTorch Lightning-friendly format. The main class is `MotionDataset` (see [data/MotionDataset.py](./data/MotionDataset.py)). To load and preprocess the data automatically, use the `MotionDataModule` (see [data/MotionDataModule.py](./data/MotionDataModule.py)). If you wish to customize data handling, please create a new file under the `data/` directory and ensure it conforms to the PyTorch Lightning framework.

### MotionDataModule Example

```python
# Setup
DATASET_PATH = "./Human Action Recognition"
BATCH_SIZE = 64
VAL_SIZE = 0.2

# Load data description
data_desc = MotionDataDescription.build_from_folder(DATASET_PATH)

# Initialize DataModule
datamodule = MotionDataModule(data_desc, batch_size=BATCH_SIZE, val_size=VAL_SIZE)
```

If you add any new data-related classes, place them in the `data/` folder. When modifying existing code, please **maintain the current API** or **add new APIs** for additional processes (recommended).

## Model

For model definitions, refer to the PyTorch Lightning documentation on building models: [https://lightning.ai/docs/pytorch/stable/starter/introduction.html](https://lightning.ai/docs/pytorch/stable/starter/introduction.html)

### Lightning Module Example

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

Add any new model classes under the `model/` directory. When modifying existing models, please **keep the current API** or **introduce new APIs** (highly recommended).

## Trainer

The `Trainer` class orchestrates the training process and supports logging and callbacks.

### Trainer Example

```python
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

LOG_PATH = "logs"
LOG_NAME = "clip_har"

# Logger setup
tb_logger = TensorBoardLogger(LOG_PATH, name=LOG_NAME)

# Checkpoint callback
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    filename="model-{epoch:02d}-{val_loss:.2f}",
    save_top_k=3,
    mode="min",
    save_last=True,
)

# Early stopping callback
early_stop_callback = EarlyStopping(
    monitor="val_loss",
    patience=10,
    mode="min",
)

# Trainer initialization
trainer = L.Trainer(
    callbacks=[checkpoint_callback, early_stop_callback],
    logger=tb_logger,
    max_epochs=EPOCH,
    log_every_n_steps=20,
    # fast_dev_run=True,
)

# Start training
trainer.fit(model, datamodule=datamodule)
```

If you want to verify the training loop quickly, set `fast_dev_run=True`. Once validated, disable it by setting `fast_dev_run=False` or commenting out the line.

> To monitor training in real-time, run:
>
> ```bash
> tensorboard --logdir {LOG_PATH}/{LOG_NAME}
> ```
