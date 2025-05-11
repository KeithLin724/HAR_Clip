import torch
import pandas as pd
from PIL import Image as ImagePIL
from torch.utils.data import Dataset
from transformers import CLIPProcessor

DEFAULT_MODEL = "openai/clip-vit-large-patch14-336"

DEFAULT_MAPPING = {
    "calling": "A photo of a person making a phone call",
    "clapping": "A photo of a person enthusiastically clapping their hands",
    "cycling": "A photo of a person riding a bicycle outdoors",
    "dancing": "A photo of a person dancing with expressive movement",
    "drinking": "A photo of a person drinking a beverage",
    "eating": "A photo of a person eating a meal",
    "fighting": "A photo of two people fighting or engaging in a physical altercation",
    "hugging": "A photo of two people hugging each other",
    "laughing": "A photo of a person laughing happily",
    "listening_to_music": "A photo of a person wearing headphones and listening to music",
    "running": "A photo of a person running at a steady pace",
    "sitting": "A photo of a person sitting on a chair or bench",
    "sleeping": "A photo of a person sleeping peacefully",
    "texting": "A photo of a person texting on a smartphone",
    "using_laptop": "A photo of a person using a laptop",
}


class MotionDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        label_dict: dict[str, int],
        processor: CLIPProcessor = None,
    ):
        self.data = data
        self.label_dict = label_dict

        self.processor = (
            processor
            if processor is not None
            else CLIPProcessor.from_pretrained(DEFAULT_MODEL)
        )

        self.process_func = (
            self.clip_preprocess if processor is None else self.open_clip_preprocess
        )
        return

    def __len__(self):
        return len(self.data)

    def clip_preprocess(self, image):
        inputs = self.processor(images=image, return_tensors="pt", padding=True)
        return inputs["pixel_values"].squeeze(0)

    def open_clip_preprocess(self, image):
        return self.processor(image)

    def __getitem__(self, idx: int):
        data_item = self.data.iloc[idx]

        filename, label = data_item["filename"], data_item["label"]
        image = ImagePIL.open(filename)

        images = self.process_func(image)
        labels = torch.tensor(self.label_dict[label])

        return images, labels


class MotionTestDataset(Dataset):
    def __init__(
        self,
        test_data: list[str],
        processor: CLIPProcessor = None,
    ):
        self.data = test_data

        self.processor = (
            processor
            if processor is not None
            else CLIPProcessor.from_pretrained(DEFAULT_MODEL)
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        data_item = self.data[idx]

        filename = data_item
        image = ImagePIL.open(filename)

        inputs = self.processor(images=image, return_tensors="pt", padding=True)

        images = inputs["pixel_values"].squeeze(0)

        return images
