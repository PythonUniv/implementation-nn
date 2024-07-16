import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from model import DiffusionModel, DenoisingDiffusionConfig, Diffusion, DDPMScheduler


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def visualize(
    diffusion: Diffusion, num_images: int = 10,
    save_points: tuple[int] = (999, 900, 850, 800, 700, 600, 400, 300, 200, 100, 50, 10, 5, 3, 0)
):
    save_points = sorted(save_points, reverse=True)
    images = diffusion(num_images=num_images, save_points=save_points, auto_cast=False)
    figure = plt.figure(figsize=(15, 15))
    axis = figure.subplots(num_images, len(save_points))
    for idx, (time_idx, tensor) in enumerate(images):
        for y, image_torch in enumerate(tensor):
            ax = axis[y, idx]
            ax.grid(False)
            image_np: np.ndarray = image_torch.permute(1, 2, 0).cpu().numpy()
            if y == 0 and idx == 3:
                print(image_np)
            image_np = np.nan_to_num(image_np, nan=0)
            image_np = image_np.clip(0, 1).astype(np.float32)
            ax.imshow(image_np)
            ax.text(0, 0, str(time_idx))
    figure.show()
    figure.savefig(os.path.join(os.path.dirname(__file__), 'diffusion.png'))


if __name__ == '__main__':
    state = torch.load(os.path.join(os.path.dirname(__file__), 'best_model_18.pt'))
    model = state['model'].cuda()
    config = state['config']
    ddpm_scheduler = DDPMScheduler(model.config).to('cuda')
    diffusion = Diffusion(model, ddpm_scheduler)
    
    visualize(diffusion, num_images=15)
