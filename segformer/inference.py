from PIL.Image import Image
import numpy as np
import torch

from segformer import SegFormer
from preprocess import SegFormerPreprocessor


class SegFormerInference:
    def __init__(self, checkpoint: str, device: torch.DeviceObjType):
        loaded = torch.load(checkpoint, device)
        self.config = loaded['config']
        self.segformer = SegFormer(self.config)
        self.segformer.load_state_dict(loaded['state_dict'])
        self.segformer.eval()
        self.preprocessor = SegFormerPreprocessor(self.config.image_size, self.config.image_size)
        
    def __call__(self, image: Image | np.ndarray) -> np.ndarray:
        preprocessed_image = self.preprocessor(image)[0].unsqueeze(dim=0)
        with torch.no_grad():
            logits = self.segformer(preprocessed_image)
        segmentation_mask = torch.softmax(logits, dim=1)
        return segmentation_mask[0].permute(1, 2, 0).cpu().numpy()
