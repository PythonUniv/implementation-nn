import os
import torch
import tqdm
from pathlib import Path
from torch.utils.data import random_split, DataLoader

from segformer import SegFormer, SegFormerConfig
from dataset import SegmentationDatasetGPUEfficient


def train(
    model: SegFormer, dataset_dir: str, batch_size: int, image_size: int = 224,
    lr: float = 5e-4, epochs: int = 1, device: torch.DeviceObjType = 'cuda', save_dir: str | None = None, processes: int | None = None,
    neptune_run=None
) -> SegFormer:
    original_images_dir = os.path.join(dataset_dir, 'original_images')
    label_images_dir = os.path.join(dataset_dir, 'label_images_semantic')
    segmentation_dataset = SegmentationDatasetGPUEfficient(original_images_dir, label_images_dir, image_size)
    train_dataset, val_dataset = random_split(segmentation_dataset, [0.8, 0.2])
    processes = processes or max(1, os.cpu_count() // 2)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=processes)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=True, num_workers=processes)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, lr, epochs=epochs, steps_per_epoch=len(train_loader), anneal_strategy='linear')
    loss_fn = torch.nn.CrossEntropyLoss()
    
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        print(epoch + 1)
        tqdm_loader = tqdm.tqdm(train_loader)
        for images, ground_truths in tqdm_loader:
            images, ground_truths = images.to(device), ground_truths.to(device)
            out = model(images)
            ground_truths = ground_truths[:, ::4, ::4]
            loss = loss_fn(out.permute(0, 2, 3, 1).view(-1, model.config.num_classes), ground_truths.contiguous().view(-1))
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            tqdm_loader.set_description(f'Loss: {loss.item():.3f}')
            
            if neptune_run is not None:
                neptune_run['train/loss'].append(loss.item())
                neptune_run['lr'].append(optimizer.param_groups[0]['lr'])

        tqdm_loader = tqdm.tqdm(val_loader)
        model.eval()
        with torch.no_grad():
            for images, ground_truths in val_loader:
                images, ground_truths = images.to(device), ground_truths.to(device)
                out = model(images)
                ground_truths = ground_truths[:, ::4, ::4]
                loss = loss_fn(out.permute(0, 2, 3, 1).view(-1, model.config.num_classes), ground_truths.contiguous().view(-1))
                tqdm_loader.set_description(f'Loss: {loss.item():.3f}')
                
                if neptune_run is not None:
                    neptune_run['val/loss'].append(loss.item())
        
    directory = Path(save_dir or os.path.join(os.path.dirname(__file__), 'checkpoints'))
    directory.mkdir(exist_ok=True)
    path = os.path.join(directory, 'model.pt')
    torch.save({'state_dict': model.state_dict(), 'config': model.config}, path)
    print('Saved.')
    return model


if __name__ == '__main__':
    import neptune
    import dotenv
    
    env_values = dotenv.dotenv_values()
    neptune_api_key = env_values['NEPTUNE_API_KEY']
    
    config = SegFormerConfig()
    model = SegFormer(config)
    
    neptune_run = neptune.init_run(project='fireml/SegFormer-train', api_token=neptune_api_key)
    
    train(model, r'C:\Users\Ноутбук\Desktop\enviroment\segformer\dataset', 16, neptune_run=neptune_run)
    print('Trained.')
