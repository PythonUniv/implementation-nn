import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertModel, DistilBertConfig, AutoTokenizer
from transformers import ViTModel, ViTConfig, AutoImageProcessor
from torch.utils.data import DataLoader
from itertools import chain
from tqdm import tqdm
from PIL import Image


class DistilBertEncoder(nn.Module):
    def __init__(self, config: DistilBertConfig | None = None):
        '''
            Init text encoder.
            
            If config is None, loading pretrained model.
        '''
        
        super().__init__()
        
        if config is None:
            self.model = DistilBertModel.from_pretrained('distilbert/distilbert-base-uncased')
        else:
            self.model = DistilBertModel._from_config(config)
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert/distilbert-base-uncased')
        
    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = self.model(x, attention_mask)
        return x
    
    
class VisionTransformerEncoder(nn.Module):
    def __init__(self, config: ViTConfig | None = None):
        super().__init__()
        
        if config is None:
            self.model = ViTModel.from_pretrained('google/vit-base-patch16-224')
        else:
            self.model = ViTModel(config)
        self.image_processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224', use_fast=True)
        
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = self.model(pixel_values=pixel_values).last_hidden_state
        x = x[:, 0]
        return x
        
        
class CLIPProjection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0):
        super().__init__()
        
        self.fc_1 = nn.Linear(in_dim, out_dim)
        self.fc_2 = nn.Linear(out_dim, out_dim)
        self.gelu = nn.GELU()
        self.layer_norm = nn.LayerNorm(out_dim, eps=1e-8)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = self.fc_1(x)
        x = self.gelu(proj)
        x = self.dropout(x)
        x = self.fc_2(x)
        x = x + proj
        x = self.layer_norm(x)
        return x
    

class CLIP(nn.Module):
    def __init__(
        self, text_encoder: nn.Module, vision_encoder: nn.Module,
        text_encoder_dim: int, vision_encoder_dim: int, proj_dim: int,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.text_encoder = text_encoder
        self.vision_encoder = vision_encoder
        
        self.text_proj = CLIPProjection(text_encoder_dim, proj_dim, dropout)
        self.vision_proj = CLIPProjection(vision_encoder_dim, proj_dim, dropout)
        
    def forward(
        self, tokens: torch.Tensor, images: torch.Tensor,
        attention_mask: torch.Tensor | None = None
    ) -> dict:
        text_encoded = self.text_encoder(tokens, attention_mask).last_hidden_state
        image_encoded = self.vision_encoder(pixel_values=images)

        text_proj = self.text_proj(text_encoded)
        vision_proj = self.vision_proj(image_encoded)
        
        similarity = text_proj @ vision_proj.T
        return {
            'text_proj': text_proj,
            'vision_proj': vision_proj,
            'similarity': similarity
        }


class CLIPInference:
    """
        Wrapper CLIP model for inference.
    """
    
    def __init__(self, clip: CLIP, device: torch.DeviceObjType):
        self.clip = clip
        self.device = device
        self.clip.to(device=device)
        self.clip.eval()
            
    @staticmethod
    def from_pretrained(path: str) -> 'CLIPInference':
        clip = torch.load(path)
        return clip
    
    def preprocess_image(self, images: torch.Tensor) -> torch.Tensor:
        processed = self.clip.vision_encoder.image_processor(images)['pixel_values']
        return processed
    
    @torch.no_grad
    def __call__(
        self, text: str | list[str] | None = None, images: list[str] | torch.Tensor | None = None,
        batch_size: int = 1, *, text_proj: torch.Tensor | None = None, image_proj: torch.Tensor | None = None
    ) -> tuple[list, list]:
        
        sim_size = (1 if isinstance(text, str) else len(text), len(images))
        similarity = torch.empty(size=sim_size, dtype=torch.float16, device=self.device)
        
        if text_proj is None:
            if isinstance(text, str):
                text = [text]
            tokenized = self.clip.text_encoder.tokenizer(text, padding=True)
            tokens = torch.tensor(tokenized['input_ids'], dtype=torch.long, device=self.device)
            attention_mask = torch.tensor(tokenized['attention_mask'], dtype=torch.long, device=self.device)
            text_encoded = self.clip.text_encoder(tokens, attention_mask)
            text_proj = self.clip.text_proj(text_encoded)
        
        if image_proj is None:
            image_encoded = self.clip.vision_encoder()
            
            for idx in range(0, len(images), batch_size):
                if isinstance(images[0], str):
                    images_batch = self.read_images(images[idx: idx + batch_size])
                else:
                    images_batch = images[idx: idx + batch_size]
                processed = self.preprocess_image(images_batch)
                image_encoded = self.clip.vision_encoder(processed)
                vision_proj = self.clip.vision_proj(image_encoded)

                similarity[:, idx: idx + batch_size] = text_proj @ vision_proj.T
        else:
            similarity = text_proj @ image_proj.T
        similarity = similarity.softmax(dim=-1).detach()
        
        ids = similarity.argsort(dim=-1)
        return similarity.tolist(), ids.tolist()
    
    @staticmethod
    def read_images(images: list[str]) -> list[Image.Image]:
        return [Image.open(path) for path in images]


def cross_entropy(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    loss = (-targets * F.log_softmax(preds)).sum(1)
    return loss


def train(
    model: CLIP, train_loader: DataLoader, val_loader: DataLoader | None = None,
    epochs: int = 1, lr: float = 1e-3, temperature: float = 1,
    neptune_run=None, device='cuda', logging_file: str = 'logs', checkpoint_dir: str | None = None
) -> CLIP:
    model.to(device)
    
    if neptune_run is not None:
        neptune_run['epochs'] = epochs
        neptune_run['lr'] = lr
        neptune_run['temperature'] = temperature
    
    with open(logging_file, 'w') as file:
        file.write(f'Epochs: {epochs} | Learning Rate: {lr}')
        
    params = [
        {'params': model.text_encoder.parameters(), 'lr': 1e-5},
        {'params': model.vision_encoder.parameters(), 'lr': 1e-5},
        {'params': chain(model.text_proj.parameters(), model.vision_proj.parameters()), 'lr': lr, 'weight_decay': 1e-3}
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=0.001, fused=True)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, lr, epochs=epochs, steps_per_epoch=len(train_loader))
    
    checkpoint_dir = checkpoint_dir or os.path.join(os.path.dirname(__file__), 'checkpoints')
    
    for epoch in range(epochs):
        print(f'Epoch: {epoch}')
        model.train()

        losses = []
        tqdm_iter = tqdm(train_loader)
        for idx, batch in enumerate(tqdm_iter):
            optimizer.zero_grad()
            
            images = batch['image']
            processed_images = model.vision_encoder.image_processor(images, do_rescale=False)['pixel_values'].to(device)
            captions = batch['caption']
            
            tokenized = model.text_encoder.tokenizer(captions, padding=True)
            tokens = torch.tensor(tokenized['input_ids'], dtype=torch.long, device=device)
            attention_mask = torch.tensor(tokenized['attention_mask'], dtype=torch.long, device=device)         
            
            output = model(tokens, processed_images, attention_mask)

            image_proj = output['image_proj']
            image_similarity = image_proj @ image_proj.T
            
            text_proj = output['text_proj']
            text_similarity = text_proj @ text_proj.T
            
            targets = F.softmax((image_similarity + text_similarity) / 2 * temperature, dim=-1)
            targets = targets.detach()
            
            similarity = output['similarity']
            
            loss = (cross_entropy(similarity, targets).sum() + cross_entropy(similarity.T, targets).sum()) / 2
            loss.backward()
        
            optimizer.step()
            scheduler.step()
            
            losses.append(loss.item())
            average_10_last_loss = sum(losses[-10:]) / len(losses[-10:])
            tqdm_iter.set_description(f'Current loss: {loss.item():.3f} | Average of last 10 batches: {average_10_last_loss:.3f}')
            
            text_encoder_lr = optimizer.param_groups[0]['lr']
            vision_encoder_lr = optimizer.param_groups[1]['lr']
            proj_lr = optimizer.param_groups[2]['lr']
            
            with open(logging_file, mode='+a') as file:
                file.write(
                    f'''Epoch: {epoch + 1} | Batch: {idx + 1} | Loss: {loss.item():.3f} | Average loss of 10 batches: {average_10_last_loss:.3f}
                        Text encoder lr: {text_encoder_lr:.3e} | Vision encoder lr: {vision_encoder_lr:.3e} | Projection lr: {proj_lr:.3e}\n\n\n
                    ''')

            if neptune_run is not None:
                neptune_run['train/loss'].append(loss.item())
                neptune_run['train/text_encoder_lr'].append(text_encoder_lr)
                neptune_run['train/vision_encoder_lr'].append(vision_encoder_lr)
                neptune_run['train/projection_lr'].append(proj_lr)

            best_loss = None
            if val_loader is not None:
                tqdm_iter = tqdm(val_loader)
                losses = []
                with torch.no_grad():
                    model.eval()
                    for batch in tqdm_iter:
                        images = batch['image']
                        processed_images = model.vision_encoder.image_processor(images, do_rescale=False)['pixel_values'].to(device)
                        
                        captions = batch['captions']
                        tokenized = model.text_encoder.tokenizer(captions, padding=True)
                        
                        tokens = torch.tensor(tokenized['input_ids'], dtype=torch.long, device=device)
                        attention_mask = torch.tensor(tokenized['attention_mask'], dtype=torch.long, device=device)
                        
                        output = model(tokens, images, attention_mask)
                        
                        image_proj = output['image_proj']
                        image_similarity = image_proj @ image_proj.T
                        
                        text_proj = output['text_proj']
                        text_similarity = text_proj @ text_proj.T
                        
                        targets = F.softmax((image_similarity + text_similarity) / 2, dim=-1)
                        
                        similarity = output['similarity']

                        loss = (cross_entropy(similarity, targets) + cross_entropy(similarity.T, targets)) / 2
                        losses.append(loss.item())
                        
                        average_loss = sum(losses) / len(losses)
                        
                        tqdm_iter.set_description(f'Loss: {loss.item():.3f} | Average loss: {average_loss:.3f}')

                        with open(logging_file, '+a') as file:
                            file.write(f'Validation | Loss: {loss.item():.3f} | Average loss: {average_loss:.3f}\n')
                        
                        if neptune_run is not None:
                            neptune_run['val/loss'].append(loss.item())
                            
                average_loss = sum(losses) / len(losses)
                if best_loss is None or average_loss < best_loss:
                    best_loss = average_loss
                    save_path = os.path.join(checkpoint_dir, f'clip_{epoch}.pt')
                    torch.save(model, save_path)
            if epoch == epochs - 1:
                save_path = os.path.join(checkpoint_dir, 'clip_last.pt')
                torch.save(model, save_path)
