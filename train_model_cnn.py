# %%
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import time
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, Callback

from model import CNNHARModel
from model import ClipHARModel
from data import MotionDataDescription, MotionDataModule

import torch

torch.set_float32_matmul_precision("high")

class TrainingMetricsLogger(Callback):
    def __init__(self):
        self.start_time = None

    def on_train_start(self, trainer, pl_module):
        self.start_time = time.time()
        print("Training started...")

    def on_train_end(self, trainer, pl_module):
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        print(f"Training completed in {elapsed_time:.2f} seconds.")

        metrics = {"elapsed_time": elapsed_time}
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
            memory_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)    # MB
            print(f"Max GPU memory allocated: {memory_allocated:.2f} MB")
            print(f"Max GPU memory reserved: {memory_reserved:.2f} MB")
            metrics["max_gpu_memory_allocated"] = memory_allocated
            metrics["max_gpu_memory_reserved"] = memory_reserved

        if trainer.logger is not None:
            trainer.logger.log_metrics(metrics, step=trainer.global_step)


EPOCH = 10

model_name = "mobilenet_v2"
fc_hidden_dim = 512
dropout_rate_fc = 0.2
learning_rate = 0.0005
# %%
metrics_logger = TrainingMetricsLogger()

# %%
data_decs = MotionDataDescription.build_from_folder("./Human Action Recognition")

# %%
datamodule = MotionDataModule(data_decs, batch_size=32, val_size=0.2)

# %% 設定model
model_cnn = CNNHARModel(model_name = model_name ,labels=data_decs.label,fc_hidden_dim = fc_hidden_dim ,dropout_rate_fc = dropout_rate_fc,learning_rate = learning_rate)
model_Clip = ClipHARModel(labels=data_decs.label)

# %%
print(model_cnn)
print(model_Clip)

# %%
cnn_version_name = f"{model_name}_fc{fc_hidden_dim}_drop{dropout_rate_fc}_lr{learning_rate}"
tb_logger_cnn = TensorBoardLogger("logs", name="cnn_har", version=cnn_version_name)
tb_logger_clip = TensorBoardLogger("logs", name="clip_har")

# %%

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    # dirpath="checkpoints",
    filename="model-{hparams.model_name}-{epoch:02d}-{val_loss:.2f}",
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
print(" Training CNN model...")
# %% 先訓練CNN
trainer_cnn = L.Trainer(
    accelerator="gpu",
    devices=1,
    callbacks=[checkpoint_callback, early_stop_callback, metrics_logger],
    logger=tb_logger_cnn,
    max_epochs=EPOCH,
    log_every_n_steps=20,
    # fast_dev_run=True,
)

# %%
print(" Training Clip model...")
trainer_clip = L.Trainer(
    accelerator="gpu",
    devices=1,
    callbacks=[checkpoint_callback, early_stop_callback, metrics_logger],
    logger=tb_logger_clip,
    max_epochs=EPOCH,
    log_every_n_steps=20,
    # fast_dev_run=True,
)
# %%
trainer_cnn.fit(model_cnn, datamodule=datamodule)

# %%
# trainer_clip.fit(model_Clip, datamodule=datamodule)
