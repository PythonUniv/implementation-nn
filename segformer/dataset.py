import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from preprocess import SegFormerPreprocessor


class SegmentationDataset(Dataset):
    def __init__(self, original_images_dir: str, label_images_dir: str, image_size: int = 224):
        self.images = sorted(os.path.join(original_images_dir, name) for name in os.listdir(original_images_dir))
        self.ground_truths = sorted(os.path.join(label_images_dir, name) for name in os.listdir(label_images_dir))
        self.preprocessor = SegFormerPreprocessor(image_size, image_size)
        self.preprocessed = None
        
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = np.array(Image.open(self.images[index]))
        ground_truth = np.array(Image.open(self.ground_truths[index]))
        
        image, ground_truth = self.preprocessor(image, ground_truth, inference=True) 
        return image, ground_truth
    
    
class SegmentationDatasetGPUEfficient(Dataset):
    def __init__(self, original_images_dir: str, label_images_dir: str, image_size: int = 224):
        self.segmentation_dataset = SegmentationDataset(original_images_dir, label_images_dir, image_size)
        self.preprocessed = []
        self.preprocess()
        
    def preprocess(self):
        data_loader = DataLoader(self.segmentation_dataset, batch_size=128, shuffle=False, num_workers=max(1, os.cpu_count() // 2))
        for images, ground_truths in data_loader:
            for image, ground_truth in zip(images, ground_truths):
                self.preprocessed.append((image, ground_truth))
    
    def __len__(self) -> int:
        return len(self.segmentation_dataset)
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.preprocessed[index]
