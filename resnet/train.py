import torch
from torch.utils.data import DataLoader
from torchvision.datasets.cifar import CIFAR100
from torchvision.transforms import ToTensor, Resize, AutoAugment, Compose, Normalize
from pathlib import Path
from tqdm import tqdm

from resnet import Resnet


def train_resnet(resnet: Resnet, batch_size: int, epochs: int, lr: float = 1e-3, device='cuda'):
    dataset_root = Path(__file__).parent / 'dataset'
    dataset_root.mkdir(exist_ok=True)
    
    save_root = Path(__file__).parent / 'save'
    save_root.mkdir(exist_ok=True)
    
    image_net_normalize = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    
    
    compose_train = Compose(
        [
            AutoAugment(),
            ToTensor(),
            Resize((224, 224)),
            image_net_normalize
        ]
    )
    
    compose_valid = Compose(
        [
            ToTensor(),
            Resize((224, 224)),
            image_net_normalize
        ]
    )
    
    train_dataset = CIFAR100(dataset_root, transform=compose_train)
    valid_dataset = CIFAR100(dataset_root, train=False, transform=compose_valid)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(resnet.parameters(), lr=lr)
    total_iters = epochs * len(train_dataloader)
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1, end_factor=0.1, total_iters=total_iters)
    
    resnet.to(device)

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}')
        train_data = tqdm(train_dataloader)
        resnet.train()
        
        for x, y in train_data:
            x = x.to(device)
            y = y.to(device)
            
            resnet_output = resnet(x)
            loss: torch.Tensor = loss_func(resnet_output, y)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            train_data.set_description(f'Loss: {loss.item():.3f}')
        
        valid_data = tqdm(valid_dataloader)
        for x, y in valid_data:
            resnet.eval()
            x = x.to(device)
            y = y.to(device)
            resnet_output = resnet(x)
            
            loss: torch.Tensor = loss_func(resnet_output, y)
            valid_data.set_description(f'Loss: {loss.item():.3f}')
        
    torch.save(resnet, save_root)
