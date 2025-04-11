import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from natsort import natsorted


@dataclass(slots=True)
class MotionDataDescription:
    train_data: pd.DataFrame
    test_data: list[str]
    label: list[str]
    label_value: dict[str, int]
    train_data_plus: pd.DataFrame = None

    @staticmethod
    def build_from_folder(folder: str, with_plus: bool = False):
        folder_name = folder
        folder: Path = Path(folder_name)

        if not folder.exists():
            raise FileNotFoundError(f"Folder {folder} does not exist")

        # preprocess the train data
        csv_file = list(folder.glob("*.csv"))[0]
        train_df = pd.read_csv(csv_file)

        mapping_path = lambda x: folder / "train_images" / x
        train_df["filename"] = train_df["filename"].apply(mapping_path)

        if with_plus:
            folder_plus = Path(f"{folder_name}_plus")
            csv_file_plus = list(folder_plus.glob("*.csv"))[0]
            train_df_plus = pd.read_csv(csv_file_plus)
            mapping_path = lambda x: folder_plus / "train_images" / x
            train_df_plus["filename"] = train_df_plus["filename"].apply(mapping_path)
        else:
            train_df_plus = None

        # list the test data
        test_data = (folder / "test_images").glob("*.jpg")
        test_data = natsorted(list(test_data))

        label = natsorted(train_df["label"].unique())

        label_dict = {label: idx for idx, label in enumerate(label)}

        return MotionDataDescription(
            train_data=train_df,
            test_data=test_data,
            label=label,
            label_value=label_dict,
            train_data_plus=train_df_plus,
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

        if self.train_data_plus is not None:
            train_data = self.train_data_plus

        return train_data, val_data
