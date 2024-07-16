import os
from dataclasses import dataclass
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader
import torchvision
from statistics import mean
import tqdm

from model import DiffusionModel, DenoisingDiffusionConfig, DDPMScheduler


@dataclass
class TrainConfig:
    epochs: int
    batch_size: int
    lr: float
    dataset_root: str
    save_path: str | None = None
    device: str = 'cuda'
    device_type: str = 'cuda'
    auto_cast: bool = True
    
    
def train(model: DiffusionModel, train_config: TrainConfig):
    transform = torchvision.transforms.ToTensor()
    train_dataset = torchvision.datasets.CIFAR100(root=train_config.dataset_root, transform=transform, download=True)
    val_dataset = torchvision.datasets.CIFAR100(root=train_config.dataset_root, train=False, transform=transform, download=True)
    use_processes = os.cpu_count() // 2
    train_data_loader = DataLoader(
        train_dataset, batch_size=train_config.batch_size, num_workers=use_processes, shuffle=True)
    val_data_loader = DataLoader(
        val_dataset, batch_size=train_config.batch_size, num_workers=use_processes, shuffle=True)
    
    model.to(train_config.device)
    ddpm_scheduler = DDPMScheduler(model.config).to(train_config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.lr)
    print(f'Number of model parameters: {sum(parameters.numel() for parameters in model.parameters()):,}')
        
    total_steps = train_config.epochs * len(train_data_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, train_config.lr, total_steps=total_steps, anneal_strategy='linear')
    
    loss_fn = torch.nn.MSELoss()
    
    best_average_loss = None
    for epoch in range(train_config.epochs):
        print(f'Starting epoch: {epoch + 1}')
        data_tqdm = tqdm.tqdm(train_data_loader)
        losses = []
        model.train()
        print('Training')
        for x, y in data_tqdm:
            x = x.to(train_config.device)
            optimizer.zero_grad()
            with torch.autocast(train_config.device_type, enabled=train_config.auto_cast):
                batch_size = x.size(0)
                time = torch.randint(0, model.config.time_steps, size=(batch_size,), device=train_config.device)
                noise = torch.randn_like(x, requires_grad=False)
                alpha = ddpm_scheduler.alpha(time).to(train_config.device)
                x = torch.sqrt(alpha) * x + torch.sqrt(1 - alpha) * noise
                output = model(x, time)
                loss = loss_fn(output, noise)
                losses.append(loss.item())
            loss.backward()
            clip_grad_norm_(model.parameters(), 1)
            data_tqdm.set_description(f'Loss on batch: {loss.item():.3f} | learning rate: {optimizer.param_groups[0]["lr"]:.3e}')
            optimizer.step()
            scheduler.step()
        average_loss = mean(losses)
        print(f'Average loss for epoch: {average_loss:.3f}')

        print('Validation')
        with torch.no_grad():
            model.eval()
            losses = []
            data_tqdm = tqdm.tqdm(val_data_loader)
            for x, y in data_tqdm:
                x = x.to(train_config.device)
                with torch.autocast(train_config.device_type, enabled=train_config.auto_cast):
                    batch_size = x.size(0)
                    time = torch.randint(0, model.config.time_steps, size=(batch_size,), device=train_config.device)
                    alpha = ddpm_scheduler.alpha(time)
                    noise = torch.randn_like(x, requires_grad=False)
                    x = torch.sqrt(alpha) * x + torch.sqrt(1 - alpha) * noise
                    output = model(x, time)
                    loss = loss_fn(output, noise)
                    losses.append(loss.item())
                    data_tqdm.set_description(f'Validation loss: {loss.item():.3f}')
            average_loss = mean(losses)
            print(f'Average loss for epoch {epoch + 1}: {average_loss:.3f}')
            
            if best_average_loss is None or average_loss < best_average_loss:
                state_dict = {
                    'model': model,
                    'config': model.config
                }
                torch.save(state_dict, os.path.join(
                    train_config.save_path or os.path.dirname(__file__), f'best_model_{epoch + 1}.pt'))
                print(f'Saved model on {epoch + 1} epoch: validation loss: {average_loss:.3f}')
                best_average_loss = average_loss
