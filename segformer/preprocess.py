import numpy as np
import torch
import albumentations as A


class SegFormerPreprocessor:
    def __init__(self, height: int, width: int):
        self.train_preprocessor = A.Compose(
            [A.Resize(height, width), A.Normalize(), A.HorizontalFlip()])
        self.inference_preprocessor = A.Compose([A.Resize(height, width), A.Normalize()])
        
    def __call__(
        self, image: np.ndarray, mask: np.ndarray | None = None,
        inference: bool = True, to_torch: bool = True
    ):
        args = {'image': image}
        if mask is not None:
            args['mask'] = mask
        if inference:
            preprocessed = self.inference_preprocessor(**args)
        else:
            preprocessed = self.train_preprocessor(**args)
        image = preprocessed.get('image')
        mask = preprocessed.get('mask')
        if to_torch:
            image = torch.tensor(image)
            image = image.permute(2, 0, 1)
            if mask is not None:
                mask = torch.tensor(mask)
        return image, mask
