import lightning as L
from .MotionDataset import MotionDataset
from torch.utils.data import DataLoader
from transformers import CLIPProcessor
from .MotionDataDescription import MotionDataDescription


class MotionDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_desc: MotionDataDescription,
        batch_size: int = 32,
        val_size: float = 0.2,
        processor: CLIPProcessor = None,
        num_workers: int = 4,
        pin_memory: bool = True,
        for_fine_tuning: bool = False,
    ):
        """
        Initialize the MotionDataModule with data specifications and training parameters.

        Parameters:
            data_desc (MotionDataDescription): Object describing the motion data to be used.
            batch_size (int, optional): Number of samples per batch. Defaults to 32.
            val_size (float, optional): Fraction of the dataset to reserve for validation. Defaults to 0.2.
            processor (CLIPProcessor, optional): Processor for data pre-processing, if applicable. Defaults to None.
            num_workers (int, optional): Number of subprocesses to utilize for data loading. Defaults to 4.
            pin_memory (bool, optional): If True, will enable memory pinning for faster data transfers (if using CUDA). Defaults to True.

        Returns:
            None
        """

        super().__init__()
        self.data_desc = data_desc
        self.batch_size = batch_size
        self.val_size = val_size
        self.processor = processor

        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.for_fine_tuning = for_fine_tuning

        return

    def setup(self, stage):
        train_df, val_df = self.data_desc.split_train_val(val_size=self.val_size)

        self.train_dataset = MotionDataset(
            data=train_df,
            label_dict=self.data_desc.label_value,
            processor=self.processor,
            for_fine_tuning=self.for_fine_tuning,
        )
        self.val_dataset = MotionDataset(
            data=val_df,
            label_dict=self.data_desc.label_value,
            processor=self.processor,
            for_fine_tuning=self.for_fine_tuning,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
