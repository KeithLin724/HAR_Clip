import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from natsort import natsorted

DEFAULT_CSV = {
    "train": "Training_set.csv",
    "test": "Testing_set.csv",
}


@dataclass(slots=True)
class MotionDataDescription:
    train_data: pd.DataFrame
    test_data: list[str]
    label: list[str]
    label_value: dict[str, int]

    @staticmethod
    def build_from_folder(folder: str):
        folder_name = folder
        folder: Path = Path(folder_name)

        if not folder.exists():
            raise FileNotFoundError(f"Folder {folder} does not exist")

        # preprocess the train data
        train_df = pd.read_csv(folder / DEFAULT_CSV["train"])

        mapping_path = lambda x: folder / "train" / x
        train_df["filename"] = train_df["filename"].apply(mapping_path)

        # list the test data
        test_data = (folder / "test").glob("*.jpg")
        test_data = natsorted(list(test_data))

        label = natsorted(train_df["label"].unique())

        label_dict = {label: idx for idx, label in enumerate(label)}

        return MotionDataDescription(
            train_data=train_df,
            test_data=test_data,
            label=label,
            label_value=label_dict,
        )

    def get_train_data_by_label(self, label: str):
        data = self.train_data[self.train_data["label"] == label]
        data = data["filename"].tolist()
        return data

    def get_label_analysis(self, normalize: bool = True) -> dict[str, float]:
        label_counts = self.train_data["label"].value_counts(normalize=normalize)
        return label_counts.to_dict()

    def split_train_val(self, val_size: float = 0.2):
        total_sample_val_size = len(self.train_data) * val_size

        labels_class = len(self.label)

        assert total_sample_val_size > labels_class, "Validation size is too small"

        each_sample_val_class_size = total_sample_val_size // labels_class

        val_indices = []
        for label in self.label:

            item_class_data = self.train_data[self.train_data["label"] == label]

            ratio_to_sample = each_sample_val_class_size / len(item_class_data)

            val_data_idx = item_class_data.sample(frac=ratio_to_sample).index.tolist()

            val_indices.extend(val_data_idx)

        val_data = self.train_data.loc[val_indices]
        train_data = self.train_data.drop(val_indices)

        return train_data, val_data
