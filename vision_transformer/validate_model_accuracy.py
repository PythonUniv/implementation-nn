import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize


from vision_transformer import VisionTransformer


def validate_model_metrics(
    model: VisionTransformer, dataset: Dataset, batch_size: int = 128,
    auto_cast: bool = True, device: str = 'cuda', device_type: str = 'cuda'
):
    val_data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    num_predictions = 0
    num_correct = 0
    
    for x, y in val_data_loader:
        with torch.no_grad(), torch.autocast(device_type, enabled=auto_cast):
            x, y = x.to(device), y.to(device)
            output: torch.Tensor = model(x)
            pred = output.argmax(dim=-1)
            
            num_predictions += pred.size(-1)
            num_correct += (pred == y).sum().item()
        print(f'Overall accuracy: {num_correct / num_predictions:.3f} | number of predictions: {num_predictions:,} | number of correct prediction: {num_correct:,}')
        
        
if __name__ == '__main__':
    import os
    from vision_transformer import ViTConfig
    from torchvision.datasets import CIFAR100
    
    config = ViTConfig()
    model = VisionTransformer(config).cuda()
    
    val_transform = Compose(
        [
            Resize(224),
            ToTensor(),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ]
    )
    
    dataset = CIFAR100(root=os.path.join(os.path.dirname(__file__), 'dataset'), transform=val_transform, train=False)
    validate_model_metrics(model, dataset)
