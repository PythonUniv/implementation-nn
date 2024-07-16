from typing import Iterable
import torch

from diffusion import DenoisingDiffusionConfig, DiffusionModel


class DDPMScheduler:
    def __init__(self, config: DenoisingDiffusionConfig):
        self.config = config
        self._beta = torch.linspace(0, 0.02, config.time_steps, dtype=torch.float32, requires_grad=False)
        self._alpha = torch.cumprod(1 - self._beta, dim=0)
        self._alpha.requires_grad_(False)
    
    def beta(self, time: torch.Tensor) -> torch.Tensor:
        batch_size = time.size(0)
        return self._beta[time].view(batch_size, 1, 1, 1)
    
    def alpha(self, time: torch.Tensor) -> torch.Tensor:
        batch_size = time.size(0)
        return self._alpha[time].view(batch_size, 1, 1, 1)
    
    def to(self, device: str) -> 'DDPMScheduler':
        self._beta = self._beta.to(device=device)
        self._alpha = self._alpha.to(device=device)
        return self


class Diffusion:
    def __init__(self, model: DiffusionModel, ddpm_scheduler: DDPMScheduler):
        self.model = model
        self.ddpm_scheduler = ddpm_scheduler
        
    @classmethod
    def from_pretrained(cls, path: str) -> 'Diffusion':
        save = torch.load(path)
        model = DiffusionModel(save['config'])
        model.load_state_dict(save['model'])
        ddpm_scheduler = DDPMScheduler(save['config'])
        return cls(model, ddpm_scheduler)
    
    def save(self, path: str):
        state_dict = {
            'model': self.model,
            'config': self.model.config
        }
        torch.save(state_dict, path)
        
    def __call__(
        self, num_images: int = 1, save_points: Iterable[int] | None = None,
        device: str = 'cuda', device_type: str = 'cuda', auto_cast: bool = True 
    ) -> list[tuple[int, torch.Tensor]]:
        self.model.eval()
        self.model.to(device)
        
        if save_points is None:
            save_points = {0}
        images = []
        with torch.no_grad(), torch.autocast(device_type, enabled=auto_cast):
            x = torch.randn(num_images, self.model.config.in_channels, *self.model.config.image_size, device=device)
            for idx in list(range(self.model.config.time_steps))[::-1]:
                time = torch.tensor(num_images * [idx], device=device)
                coef = self.ddpm_scheduler.beta(time) / \
                    (torch.sqrt(1 - self.ddpm_scheduler.alpha(time)) * torch.sqrt(1 - self.ddpm_scheduler.beta(time)))
                coef = coef.to(device=device)
                noise = self.model(x, time)
                x = x / torch.sqrt(1 - self.ddpm_scheduler.beta(time)) - coef * noise
                if idx in save_points:
                    images.append((idx, x))
                if idx != 0:
                    additional_noise = torch.randn_like(x)
                    x = x + torch.sqrt(self.ddpm_scheduler.beta(time)) * additional_noise
        return images
