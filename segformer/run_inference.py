import os
import numpy as np
from inference import SegFormerInference


def run_inference():
    segformer_inference = SegFormerInference(
        os.path.join(os.path.dirname(__file__), 'checkpoints', 'model.pt'), 'cuda')
    image = np.random.randint(low=0, high=255, size=(1024, 1024, 3), dtype=np.uint8)
    segmentation_mask = segformer_inference(image)
    print(segmentation_mask)
    
    
if __name__ == '__main__':
    run_inference()
