from os import listdir
from os.path import join

import polars as pl
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose


class ImageDataset(Dataset):
    def __init__(self, image_dir: str, metadata_path: str, transforms: Compose, convert_rgb: bool = True):
        self.metadata = pl.read_csv(metadata_path).rename({'': 'index'}).with_columns(
            file_key=pl.col('FileName_OrigRNA').str.split('-').list.first().str.reverse().str.slice(3).str.reverse())
        self.target_dict = {tgt: torch.tensor(i, dtype=torch.int64) for i, tgt in
                            enumerate(self.metadata.get_column('Metadata_pert_iname').unique())}

        self.image_dir = image_dir
        self.images = listdir(image_dir)
        self.length = len(self.images)

        self.transforms = transforms
        self.convert_rgb = convert_rgb

    def __getitem__(self, idx):
        filename = self.images[idx]
        img = Image.open(join(self.image_dir, filename))
        img = img.convert('RGB') if self.convert_rgb else img

        target = self.metadata.filter(pl.col('file_key').eq(filename.split('_')[0])).item(row=0,
                                                                                          column='Metadata_pert_iname')

        return self.transforms(img), self.target_dict[target]

    def __len__(self):
        return self.length

    def n_classes(self) -> int:
        return self.metadata.get_column('Metadata_pert_iname').n_unique()
