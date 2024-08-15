import os
from io import BytesIO
import numpy as np
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pyarrow.parquet import ParquetFile
from huggingface_hub import snapshot_download

from preprocess import SegFormerPreprocessor


def download(dir: str | None = None):
    directory = dir or os.path.join(os.path.dirname(__file__), 'dataset')
    snapshot_download('Chris1/cityscapes_segmentation', local_dir=directory, repo_type='dataset')
        
        
class SegmentationDataset(Dataset):
    def __init__(self, parquets: list[str], height: int, width: int, augmentation: bool = True):
        self.parquets = parquets
        
        self.len = sum(ParquetFile(parquet).metadata.num_rows for parquet in parquets)
        self.current_parquet_idx = None
        self.current_parquet = None
        self.current_parquet_rows: int | None = None
        self.idx = 0
        self.preprocessor = SegFormerPreprocessor(height, width)
        self.augmentation = augmentation
        self.next_parquet()
            
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.idx == self.current_parquet_rows:
            self.next_parquet()

        image = np.array(Image.open(BytesIO(self.current_parquet.iloc[self.idx]['image']['bytes'])))
        ground_truth = np.array(Image.open(BytesIO(self.current_parquet.iloc[self.idx]['semantic_segmentation']['bytes'])))[:, :, 0]
        self.idx += 1
        image, ground_truth = self.preprocessor(image, ground_truth, inference=(not self.augmentation), to_torch=True)
        return image, ground_truth
        
    def __len__(self):
        return self.len
    
    def next_parquet(self):
        next_idx = (self.current_parquet_idx + 1) % len(self.parquets) if self.current_parquet_idx is not None else 0
        self.current_parquet = pd.read_parquet(self.parquets[next_idx])
        self.current_parquet_rows = len(self.current_parquet)
        self.idx = 0

        
def get_data_loader(parquets: list[str], batch_size: int, height: int, width: int, train: bool = True, dir: str | None = None) -> DataLoader:
    directory = dir or os.path.join(os.path.dirname(__file__), 'dataset', 'data')
    parquets = [os.path.join(directory, name) for name in parquets]
    dataset = SegmentationDataset(parquets, height, width, train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return loader


if __name__ == '__main__':
    download()
