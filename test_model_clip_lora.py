# %%
import torch
from test_utils import TestRunner

from model import ClipLoRaHARModel
from data import MotionDataDescription, MotionDataModule


SEED = 42
torch.manual_seed(SEED)

torch.set_float32_matmul_precision("high")

# %%
data_decs = MotionDataDescription.build_from_folder("./Human Action Recognition")

datamodule = MotionDataModule(data_decs, batch_size=64, val_size=0.2)


# %%
model = ClipLoRaHARModel.load_from_checkpoint(
    "./logs/clip_har_lora/version_0/checkpoints/model-epoch=09-val_loss=2.02.ckpt"
)
# %%
print(model)


# %%
label_value = torch.arange(len(ClipLoRaHARModel.DEFAULT_MAPPING))
test_runner = TestRunner.test_model(
    model=model,
    dummy_input=(torch.randn(1, 3, 336, 336), label_value),
    datamodule=datamodule,
    skip_profile=True,
)
# %%
print(test_runner)

# %%
test_runner.to_folder(folder_name="test_results_clip_lora_har")

# %%
