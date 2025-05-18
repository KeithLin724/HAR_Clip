# %%
import torch
from test_utils import TestRunner

from model import ClipBaselineModel
from data import MotionDataDescription, MotionDataModule


SEED = 42
torch.manual_seed(SEED)

torch.set_float32_matmul_precision("high")

# %%
data_decs = MotionDataDescription.build_from_folder("./Human Action Recognition")

datamodule = MotionDataModule(data_decs, batch_size=64, val_size=0.2)


# %%
model = ClipBaselineModel()  # use_prompt=False is use keyword
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
# folder_name="test_results_clip_har_baseline_label_version" for use_prompt=False
test_runner.to_folder(folder_name="test_results_clip_har_baseline")

# %%
