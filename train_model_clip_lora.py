# %%
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from model import ClipLoRaHARModel, ClipLoRaConfig
from data import MotionDataDescription, MotionDataModule

import torch

SEED = 42
torch.manual_seed(SEED)

torch.set_float32_matmul_precision("high")

EPOCH = 10

# %%
data_decs = MotionDataDescription.build_from_folder("./Human Action Recognition")

# %%

datamodule = MotionDataModule(data_decs, batch_size=8, val_size=0.2)
# %%

# TODO: check other lora config
lora_config = ClipLoRaConfig(r=16, lora_alpha=32)

# TODO: add lora in different condition
build_condition = lambda name, module_x: isinstance(module_x, (torch.nn.Linear))
# build_condition = lambda name, module_x: isinstance(module_x, (torch.nn.Linear, torch.nn.Embedding))
# build_condition = lambda name, module_x: isinstance(module_x, (torch.nn.Linear, torch.nn.Embedding)) and "text_model" not in name
# build_condition = lambda name, module_x: isinstance(module_x, (torch.nn.Linear, torch.nn.Embedding)) and "vision_model" not in name


lora_config.lazy_build_target_modules(build_condition)

model = ClipLoRaHARModel.from_config(lora_config)
print(model)

# %%
# print(model)
# %%
tb_logger = TensorBoardLogger("logs", name="clip_har_lora")

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
# Initialize the Lightning Trainer with callbacks, logger, and multi-GPU support
trainer = L.Trainer(
    callbacks=[checkpoint_callback, early_stop_callback],
    logger=tb_logger,
    max_epochs=EPOCH,
    log_every_n_steps=20,
    devices=[0, 1, 2, 3],  # Use 4 GPUs (IDs 0-3)
    # fast_dev_run=True,
)


# %%
trainer.fit(model, datamodule=datamodule)

# %%
trainer.validate(model, datamodule=datamodule)
