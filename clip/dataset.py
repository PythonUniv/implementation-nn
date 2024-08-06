import random
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from datasets import DatasetDict
from torchvision.transforms import ToTensor, Resize


class ImageCaptionDataset(Dataset):
    def __init__(self, dataset: DatasetDict):
        self.dataset = dataset
        self.to_tensor = ToTensor()
        self.resize = Resize(size=(224, 224))
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index: int):
        return {
            'caption': random.choice(self.dataset[index]['caption']),
            'image': self.resize(self.to_tensor(self.dataset[index]['image']))
        }
    

def get_data_loaders(batch_size: int = 1, num_workers: int = 0) -> tuple[DataLoader, DataLoader]:
    
    hugging_face_dataset = load_dataset('nlphuji/flickr30k')['test']
    train_test_split = hugging_face_dataset.train_test_split(train_size=0.8)
    
    train_dataset = ImageCaptionDataset(train_test_split['train'])
    val_dataset = ImageCaptionDataset(train_test_split['test'])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    return train_loader, val_loader
