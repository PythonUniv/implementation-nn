import os
import torch
import tqdm
from pathlib import Path

from segformer import SegFormer, SegFormerConfig
from dataset import SegmentationDataLoader


def train(
    model: SegFormer, train_parquets: list[str], val_parquets: list[str], batch_size: int,
    lr: float = 5e-4, epochs: int = 1, device: torch.DeviceObjType = 'cuda', save_dir: str | None = None
) -> SegFormer:
    train_loader = SegmentationDataLoader(
        train_parquets, model.config.image_size, model.config.image_size, batch_size=batch_size, augmentation=True)
    val_loader = SegmentationDataLoader(
        val_parquets, model.config.image_size, model.config.image_size, batch_size=batch_size, augmentation=False)
    
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

        tqdm_loader = tqdm.tqdm(val_loader)
        model.eval()
        with torch.no_grad():
            for images, ground_truths in val_loader:
                images, ground_truths = images.to(device), ground_truths.to(device)
                out = model(images)
                ground_truths = ground_truths[:, ::4, ::4]
                loss = loss_fn(out.permute(0, 2, 3, 1).view(-1, model.config.num_classes), ground_truths.contiguous().view(-1))
                tqdm_loader.set_description(f'Loss: {loss.item():.3f}')
        
    directory = Path(save_dir or os.path.join(os.path.dirname(__file__), 'checkpoints'))
    directory.mkdir(exist_ok=True)
    path = os.path.join(directory, 'model.pt')
    torch.save({'state_dict': model.state_dict(), 'config': model.config}, path)
    print('Saved.')
    return model


if __name__ == '__main__':
    config = SegFormerConfig()
    model = SegFormer(config)
    train(
        model, ['validation-00000-of-00003.parquet', 'validation-00001-of-00003.parquet'],
        ['validation-00002-of-00003.parquet'], batch_size=16, device='cuda')
