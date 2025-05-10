# %%
import torch
from test_utils import TestRunner

from model import ClipHARModel,CNNHARModel
from data import MotionDataDescription, MotionDataModule

from lightning.pytorch.callbacks import ModelCheckpoint

SEED = 42
torch.manual_seed(SEED)

torch.set_float32_matmul_precision("high")

# %%
data_decs = MotionDataDescription.build_from_folder("./Human Action Recognition")

datamodule = MotionDataModule(data_decs, batch_size=64, val_size=0.2)


# %%
model = CNNHARModel.load_from_checkpoint(
    "./logs/cnn_har/mobilenet_v2_fc512_drop0.3_lr0.001/checkpoints/model-epoch=09-val_loss=0.82.ckpt"
)
# %%
print(model)


# %%
test_runner = TestRunner.test_model(
    model=model,
    dummy_input=torch.randn(1, 3, 336, 336),
    datamodule=datamodule,
)
# %%
print(test_runner)

# %%
test_runner.to_folder(folder_name="test_results_cnn_mobilenet_v2_har")

# %%
