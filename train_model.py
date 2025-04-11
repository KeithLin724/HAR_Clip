# %%
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from model import ClipHARModel
from data import MotionDataDescription, MotionDataModule

import torch

torch.set_float32_matmul_precision("high")


EPOCH = 10

# %%
data_decs = MotionDataDescription.build_from_folder("./Human Action Recognition")

# %%

datamodule = MotionDataModule(data_decs, batch_size=64, val_size=0.2)
# %%
model = ClipHARModel(labels=data_decs.label)
print(model)

# %%
# print(model)
# %%
tb_logger = TensorBoardLogger("logs", name="clip_har")

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    # dirpath="checkpoints",
    filename="model-{epoch:02d}-{val_loss:.2f}",
    save_top_k=3,
    mode="min",
    save_last=True,
)


early_stop_callback = EarlyStopping(
    monitor="val_loss",
    patience=10,
    mode="min",
)

# %%
trainer = L.Trainer(
    callbacks=[checkpoint_callback, early_stop_callback],
    logger=tb_logger,
    max_epochs=EPOCH,
    log_every_n_steps=20,
    # fast_dev_run=True,
)

# %%
trainer.fit(model, datamodule=datamodule)
