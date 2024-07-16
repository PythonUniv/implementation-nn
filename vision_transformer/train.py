import os
import torch
from torchvision import transforms
from torchvision.datasets import Food101
from torch.utils.data import DataLoader
import tqdm

from vision_transformer import VisionTransformer, ViTConfig


def train(
    model: VisionTransformer, image_size: int = 224, epochs: int = 3, batch_size: int = 32,
    mini_batch_size: int = 16, learning_rate: float = 5e-4, device='cuda', auto_cast: bool = True, device_type: str = 'cuda',
    dataset_root: str | None = None
):
    assert batch_size % mini_batch_size == 0
    
    image_net_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            image_net_normalize
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            image_net_normalize
        ]
    )
    
    data_root = dataset_root or os.path.join(os.path.dirname(__file__), 'dataset')
    dataset_train = Food101(data_root, split='train', transform=train_transform, download=True)
    dataset_val = Food101(data_root, split='test', transform=val_transform, download=True)
    data_loader_train = DataLoader(dataset_train, batch_size=mini_batch_size, shuffle=True)
    data_loader_val = DataLoader(dataset_val, batch_size=mini_batch_size, shuffle=True)
    
    grad_steps = batch_size // mini_batch_size
    steps_per_epoch = len(data_loader_train) // grad_steps
    total_steps = epochs * steps_per_epoch
    
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    one_cycle_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=learning_rate, total_steps=total_steps, anneal_strategy='linear')
    reduce_on_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=100)
    
    print(f'Model config: {model.config}')
    print(f'Training on {epochs:,} | batch size: {batch_size:,} | mini batch size: {mini_batch_size:,} | total steps: {total_steps:,}')
    
    for epoch in range(epochs):
        print(f'Training on {epoch} epoch.')
        
        model.train()
        train_tqdm = tqdm.tqdm(total=steps_per_epoch)
        train_iter = iter(data_loader_train)
        
        for step in range(steps_per_epoch):
            optimizer.zero_grad()
            accumulated_loss = 0
            for mini_step in range(grad_steps):
                x, y = next(train_iter)
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type, enabled=auto_cast):
                    output = model(x)
                    loss = torch.nn.functional.cross_entropy(output, y)
                    loss /= grad_steps
                loss.backward()
                accumulated_loss += loss.item()
            optimizer.step()
            one_cycle_scheduler.step()
            reduce_on_plateau.step(accumulated_loss)
            train_tqdm.update(1)
            train_tqdm.set_description(f'Train loss: {accumulated_loss:.3f} | learning rate: {optimizer.param_groups[0]["lr"]:.3e}')
        
        val_tqdm = tqdm.tqdm(total=len(data_loader_val) // grad_steps)
        val_iter = iter(data_loader_val)
        model.eval()
        with torch.no_grad():
            for step in range(len(data_loader_val) // grad_steps):
                accumulated_loss = 0
                for mini_step in range(grad_steps):
                    with torch.autocast(device_type, enabled=auto_cast):
                        x, y = next(val_iter)
                        x, y = x.to(device), y.to(device)
                        output = model(x)
                        loss = torch.nn.functional.cross_entropy(output, y)
                        accumulated_loss += loss.item() / grad_steps
                val_tqdm.update(1)
                val_tqdm.set_description(f'Validation loss: {accumulated_loss:.3f}')
            

if __name__ == '__main__':
    device = 'cuda'
    device_type = 'cuda'
    config = ViTConfig(image_size=32, patch_size=4, dim=256, num_blocks=8, num_heads=4, num_classes=100)
    model = VisionTransformer(config).to(device)
    train(model, image_size=32, epochs=3, batch_size=128, mini_batch_size=128, device=device, device_type=device_type, auto_cast=False, learning_rate=1e-3)
