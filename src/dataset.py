import json

import torch
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm


class COCODataset(Dataset):
    def __init__(self, data_dir, split, transform=None):
        self.data_dir = Path(data_dir)
        self.split = split
        assert self.split in ["train", "val"]
        self.transform = transform
        with open(Path(self.data_dir) / "annotations" / "train_caption.json", "r") as f:
            captions = json.load(f)
            # TODO: what if error?
        self.captions = self._get_split(captions)

    def _get_split(self, captions):
        data = []
        for cap in tqdm(captions):
            path = (
                self.data_dir
                / f"{self.split}2014"
                / f"COCO_{self.split}2014_{int(cap['image_id']):012d}.jpg"
            )
            if path.is_file():
                data.append(cap)
        return data

    def __getitem__(self, idx):
        caption = self.captions[idx]
        img_id = int(caption["image_id"])
        # embed_path = self.data_dir / "embeds" / self.clip_model / f"{img_id}.pt"
        # if embed_path.is_file():
        #     embed = torch.load(embed_path)
        # else:
        path = (
            self.data_dir
            / f"{self.split}2014"
            / f"COCO_{self.split}2014_{img_id:012d}.jpg"
        )
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, caption['caption']
