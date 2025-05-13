import lightning as L
import torch
from thop import profile
from dataclasses import dataclass, field, asdict
import torch.nn as nn
from torchinfo import summary, ModelStatistics
from loguru import logger
from typing import Callable

from pathlib import Path
from data import MotionDataDescription, MotionDataModule
import json


def timed(fn: Callable):
    """
    The `timed` function in Python measures the execution time of a given function using CUDA events.

    :param fn: The `fn` parameter in the `timed` function is expected to be a callable object, such as a
    function or a method, that you want to measure the execution time of. You can pass any function or
    method as an argument to the `timed` function to measure its execution time
    :type fn: Callable
    :return: The `timed` function returns a tuple containing the result of the function `fn` and the
    time taken for `fn` to execute in seconds.
    """
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000


@dataclass(slots=True)
class TestRunner:
    model_type: str
    flops: float
    params: float
    model_structure: nn.Module = field(repr=False)
    summary_out: ModelStatistics = field(repr=False)
    run_time: float
    test_result: list

    def to_folder(self, folder_name: str = None) -> Path:
        folder = (
            self.model_type.replace(".", "_") if folder_name is None else folder_name
        )
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)

        json_str = asdict(self)

        del json_str["model_structure"]
        del json_str["summary_out"]
        print(json_str)

        with open(folder / f"detail.json", "w") as f:
            json_str = json.dumps(json_str, indent=4)
            f.write(json_str)

        with open(folder / "summary.txt", "w") as f:
            f.write(str(self.summary_out))

        with open(folder / "model_structure.txt", "w") as f:
            f.write(str(self.model_structure))

        return folder

    @staticmethod
    def test_model(
        model: L.LightningModule,
        dummy_input: torch.Tensor,
        datamodule: L.LightningDataModule = None,
        trainer: L.Trainer = None,
        skip_profile: bool = False,
    ):
        logger.info(f"Testing model: {type(model)}")

        if datamodule is None:
            datamodule = TestRunner.build_test_datamodule()

        if trainer is None:
            trainer = L.Trainer()

        model.eval()
        model.freeze()

        if isinstance(dummy_input, tuple):
            dummy_input = tuple(
                [
                    x.to(model.device) if isinstance(x, torch.Tensor) else x
                    for x in dummy_input
                ]
            )
        else:
            dummy_input = dummy_input.to(model.device)

        if not skip_profile:

            logger.info("Run profile...")

            inputs = (
                (dummy_input,) if not isinstance(dummy_input, tuple) else dummy_input
            )

            flops, params = profile(model, inputs=inputs)
        else:
            logger.info("Skip profile...")
            flops, params = "Skipped", "Skipped"

        logger.info("Run summary...")
        summary_out = summary(
            model,
            input_data=dummy_input,
            col_names=(
                "input_size",
                "output_size",
                "num_params",
                "params_percent",
                "kernel_size",
                "mult_adds",
                "trainable",
            ),
            device="cuda",
        )
        output = trainer.validate(model, datamodule=datamodule)

        model.cuda()

        run_func = lambda: model(dummy_input)
        if isinstance(dummy_input, tuple):
            run_func = lambda: model(*dummy_input)

        _, run_time = timed(run_func)
        logger.info(f"Model run time: {run_time:.2f} s")

        return TestRunner(
            model_type=str(type(model)),
            flops=flops,
            params=params,
            model_structure=model,
            summary_out=summary_out,
            run_time=run_time,
            test_result=output,
        )

    @staticmethod
    def build_test_datamodule(
        folder: str = "./Human Action Recognition",
        batch_size: int = 64,
        val_size: float = 0.2,
    ):
        data_decs = MotionDataDescription.build_from_folder(folder)

        datamodule = MotionDataModule(
            data_decs, batch_size=batch_size, val_size=val_size
        )
        return datamodule
