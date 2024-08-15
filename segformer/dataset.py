import os
from io import BytesIO
import math
import numpy as np
from PIL import Image
import pandas as pd
import torch
from pyarrow.parquet import ParquetFile
from joblib import Parallel, delayed
from huggingface_hub import snapshot_download

from preprocess import SegFormerPreprocessor


def download(dir: str | None = None):
    directory = dir or os.path.join(os.path.dirname(__file__), 'dataset')
    snapshot_download('Chris1/cityscapes_segmentation', local_dir=directory, repo_type='dataset')
        
        
class SegmentationDataLoader:
    def __init__(
        self, parquets: list[str], height: int, width: int,
        batch_size: int, augmentation: bool = True, processes: int | None = None
    ):
        self.parquets = parquets
        self.height = height
        self.width = width
        self.augmentation = augmentation
        self.idx = None
        self.iterator = None
        self.batch_size = batch_size        
        self.processes = processes or max(1, os.cpu_count() // 2)
        self.preprocessor = SegFormerPreprocessor(height, width)
        self.len = sum(math.ceil(ParquetFile(parquet).metadata.num_rows / batch_size) for parquet in parquets)
        self.iters = 0
        
    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.iterator is None:
            self.load_iterator()
            
        if self.iters == self.len:
            raise StopIteration
        
        try:
            batch: pd.DataFrame = next(self.iterator).to_pandas()
        except StopIteration:
            self.load_iterator()
            batch: pd.DataFrame = next(self.iterator).to_pandas()
            
        preprocessed_images = torch.empty(self.batch_size, 3, self.height, self.width)
        preprocessed_ground_truths = torch.empty(self.batch_size, self.height, self.width, dtype=torch.long)
        preprocessed = Parallel(self.processes, backend='threading')([self.preprocess(data) for idx, data in batch.iterrows()])
        for idx, (preprocessed_image, preprocessed_ground_truth) in enumerate(preprocessed):
            preprocessed_images[idx] = preprocessed_image
            preprocessed_ground_truths[idx] = preprocessed_ground_truth
        self.iters += 1 
        return preprocessed_images, preprocessed_ground_truths
    
    def __iter__(self):
        self.iters = 0
        return self
                
    def load_iterator(self):
        self.idx = (self.idx + 1) % len(self.parquets) if self.idx is not None else 0
        parquet_path = self.parquets[self.idx]
        self.iterator = ParquetFile(parquet_path).iter_batches(self.batch_size)
        
    def __len__(self) -> int:
        return self.len

    @delayed
    def preprocess(self, data: pd.Series) -> tuple[torch.Tensor, torch.Tensor]:
        image = np.array(Image.open(BytesIO(data['image']['bytes'])))
        segmentation_mask = np.array(Image.open(BytesIO(data['semantic_segmentation']['bytes'])))[:, :, 0]
        preprocessed_image, preprocessed_ground_truth = self.preprocessor(image, segmentation_mask, inference=(not self.augmentation))
        return preprocessed_image, preprocessed_ground_truth
        
    
if __name__ == '__main__':
    download()
