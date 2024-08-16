import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from preprocess import SegFormerPreprocessor


class SegmentationDataset(Dataset):
    def __init__(self, original_images_dir: str, label_images_dir: str, image_size: int = 224):
        self.images = sorted(os.path.join(original_images_dir, name) for name in os.listdir(original_images_dir))
        self.ground_truths = sorted(os.path.join(label_images_dir, name) for name in os.listdir(label_images_dir))
        self.preprocessor = SegFormerPreprocessor(image_size, image_size)
        
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = np.array(Image.open(self.images[index]))
        ground_truth = np.array(Image.open(self.ground_truths[index]))
        
        image, ground_truth = self.preprocessor(image, ground_truth, inference=True) 
        return image, ground_truth
