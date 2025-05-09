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

model_name = ["resnet18","resnet34","mobilenet_v2"] #
fc_hidden_dim = [128,256,512]            #
dropout_rate_fc = [0.2,0.3,0.4,0.5,0.6]  #
learning_rate = [0.01,0.005,0.001,0.0005,0.0001]
# %%
metrics_logger = TrainingMetricsLogger()

# %%
data_decs = MotionDataDescription.build_from_folder("./Human Action Recognition")

# %%
datamodule = MotionDataModule(data_decs, batch_size=32, val_size=0.2)

# %% 設定model
model_Clip = ClipHARModel(labels=data_decs.label)

# %%
print(model_Clip)

# %%
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
# %% 訓練 Clip
# trainer_clip.fit(model_Clip, datamodule=datamodule)
# %% 訓練 CNN
from itertools import product

# for each combination of parameters
for m_name, fc_dim, drop_rate, lr in product(model_name, fc_hidden_dim, dropout_rate_fc, learning_rate):
    # 產生對應的版本名稱
    cnn_version_name = f"{m_name}_fc{fc_dim}_drop{drop_rate}_lr{lr}"
    print(f"Training CNN: {cnn_version_name}")

    # 建立 Logger
    tb_logger_cnn = TensorBoardLogger("logs", name="cnn_har", version=cnn_version_name)

    # 建立模型，此處只傳入單一參數，不是整個 list
    model_cnn = CNNHARModel(model_name=m_name,
                             labels=data_decs.label,
                             fc_hidden_dim=fc_dim,
                             dropout_rate_fc=drop_rate,
                             learning_rate=lr)

    # 建構 trainer
    trainer_cnn = L.Trainer(
        accelerator="gpu",
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback, metrics_logger],
        logger=tb_logger_cnn,
        max_epochs=EPOCH,
        log_every_n_steps=20,
    )

    # 開始訓練
    trainer_cnn.fit(model_cnn, datamodule=datamodule)

    # 釋放資源並清空 GPU cache
    del trainer_cnn, model_cnn
    torch.cuda.empty_cache()
