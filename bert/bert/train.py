import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tokenizers import Tokenizer, Encoding
import tqdm
import random
import itertools
from typing import Callable, Literal
from pathlib import Path
from statistics import mean

from .pretraining import BertPretrain
from .utils import flatten_2d


class BertTrainer:
    def __init__(
        self, bert_pratrain: BertPretrain, tokenizer: Tokenizer, device: str,
        special_token_ids: list[int], mask_token_id: int = 103, summary_writer: SummaryWriter | None = None
    ) -> None:
        
        self.bert_pretrain = bert_pratrain.to(device)
        self.tokenizer = tokenizer
        self.tokenizer.enable_padding()
        self.summary_writer = summary_writer
        self.device = device
        self.mask_token_id = mask_token_id
        self.special_token_ids = special_token_ids
        self.train_global_step = 0
        self.valid_global_step = 0
        self.not_special_tokens = [idx for idx in range(tokenizer.get_vocab_size()) if idx not in self.special_token_ids]
        
    def change_token(self, token: int) -> int:
        p = random.random()
        if p < 0.1:
            return token
        if p < 0.2:
            return random.choice(self.not_special_tokens)
        return self.mask_token_id
        
    def get_samples(self, sentences_1: tuple[str], sentences_2: tuple[str]) -> tuple[Tensor, ...]:
        """
            Make a training samples from sentences.
            
            Returns:
                tuple of torch.Tensor of form (unchange_sample, changed_sample, is_next, positions, attention_mask, segment_mask) 
        """
        
        middle = len(sentences_1) // 2
        next_pairs = list(zip(sentences_1[: middle], sentences_2[: middle]))
        random_pairs = list(itertools.chain(sentences_1[middle:], sentences_2[middle:]))
        not_next_pairs = [random_pairs[idx: idx + 2] for idx in range(0, len(random_pairs), 2)]
        
        encodings: list[Encoding] = self.tokenizer.encode_batch(next_pairs + not_next_pairs)
        
        y = torch.ones(len(sentences_1), dtype=torch.long)
        y[middle:] = 0
        
        x = torch.tensor([encoding.ids for encoding in encodings], dtype=torch.long)
        attention_mask = torch.tensor([encoding.attention_mask for encoding in encodings], dtype=torch.long)
        segment_mask = torch.tensor([encoding.type_ids for encoding in encodings], dtype=torch.long)
        special_tokens_mask = [encoding.special_tokens_mask for encoding in encodings]
        
        positions = [random.sample({idx for idx, is_special in enumerate(mask)
                     if not is_special}, k=int(0.15 * mask.count(0))) for mask in special_tokens_mask]
        
        x_changed = x.clone().detach()
        
        batch_ids = [idx for idx, mask_ids in enumerate(positions) for item in mask_ids]
        x_changed[batch_ids, flatten_2d(positions)].apply_(self.change_token)
        return x, x_changed, y, positions, attention_mask, segment_mask
        
    def train_on_epoch(
        self, train_data: DataLoader, valid_data: DataLoader, masked_language_loss_func: Callable,
        next_sentence_loss_func: Callable, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler,
        reduce_on_plateau_scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau, valid_period: int = 1000
    ):
        
        data = tqdm.tqdm(train_data)
        for batch_idx, (sentences_1, sentences_2) in enumerate(data):
            self.bert_pretrain.train()
            optimizer.zero_grad()
            
            x, x_changed, y, positions, attention_mask, segment_mask = self.get_samples(sentences_1, sentences_2)
            
            x = x.to(self.device)
            x_changed = x_changed.to(self.device)
            y = y.to(self.device)
            attention_mask = attention_mask.to(self.device)
            segment_mask = segment_mask.to(self.device)
            
            masked_language, next_sentence = self.bert_pretrain(x_changed, positions, segment_mask, attention_mask)
            
            mask_batch_ids = [idx for idx, mask in enumerate(positions) for item in mask]
            x_pred = x[mask_batch_ids, flatten_2d(positions)]
            masked_language_loss = masked_language_loss_func(masked_language, x_pred)
            next_sentence_loss = next_sentence_loss_func(next_sentence, y)
            loss: Tensor = masked_language_loss + next_sentence_loss
            loss.backward()
            
            optimizer.step()
            scheduler.step()
            
            masked_language_acc = torch.mean((x_pred == masked_language.argmax(dim=-1)).float())
            next_sentence_acc = torch.mean((y == next_sentence.argmax(dim=-1)).float())
            
            self.log(
                {
                    'loss': loss.item(),
                    'masked_language_loss': masked_language_loss.item(),
                    'next_sentence_loss': next_sentence_loss.item(),
                    'masked_language_acc': masked_language_acc.item(),
                    'next_sentence_acc': next_sentence_acc.item(),
                    'lr': optimizer.param_groups[0]['lr']
                },
                'train',
                self.train_global_step
            )
            self.train_global_step += 1
            
            data.set_description_str(f'Loss: {loss.item():.3f}')
            
            if (batch_idx + 1) % valid_period == 0:
                valid_loss = self.valid_on_epoch(valid_data, masked_language_loss_func, next_sentence_loss_func)
                reduce_on_plateau_scheduler.step(valid_loss)
            
    def log(self, metrics: dict, mode: Literal['train', 'valid'], global_step: int):
        if self.summary_writer is not None:
            for name, value in metrics.items():
                self.summary_writer.add_scalar(f'{mode}/{name}', value, global_step=global_step)
            
    def valid_on_epoch(self, valid_data: DataLoader, masked_language_loss_func: Callable, next_sentence_loss_func: Callable) -> float:
        self.bert_pretrain.eval()
        losses, masked_language_losses, next_sentence_losses, masked_language_accs, next_sentence_accs = [], [], [], []
        
        data = tqdm.tqdm(valid_data)
        for sentences_1, sentences_2 in data:
            x, x_changed, y, positions, attention_mask, segment_mask = self.get_samples(sentences_1, sentences_2)
            x = x.to(self.device)
            x_changed = x_changed.to(self.device)
            y = y.to(self.device)
            attention_mask = attention_mask.to(self.device)
            segment_mask = segment_mask.to(self.device)
            
            masked_language, next_sentence = self.bert_pretrain(x_changed, positions, segment_mask, attention_mask)
            batch_ids = [idx for idx, mask in enumerate(positions) for item in mask]
            x_pred = x[batch_ids, flatten_2d(positions)]
            
            masked_language_loss = masked_language_loss_func(masked_language, x_pred)
            next_sentence_loss = next_sentence_loss_func(next_sentence, y)
            loss = masked_language_loss + next_sentence
            masked_language_acc = torch.mean(x_pred == masked_language.argmax(dim-1))
            next_sentence_acc = torch.mean(y == next_sentence.argmax(dim=-1))
            
            losses.append(loss.item)
            masked_language_losses.append(masked_language_loss.item())
            next_sentence_losses.append(next_sentence_loss.item())
            masked_language_accs.append(masked_language_acc.item())
            next_sentence_accs.append(next_sentence_acc.item())
            
            self.log(
                {
                    'loss': loss.item(),
                    'masked_language_loss': masked_language_loss.item(),
                    'next_sentence_loss': next_sentence_loss.item(),
                    'masked_language_acc': masked_language_acc.item(),
                    'next_sentence_acc': next_sentence_acc.item()
                },
                'valid',
                global_step=self.valid_global_step
            )
            self.valid_global_step += 1
            
            data.set_description(f'Loss: {loss.item():.3f}')
            
        print(
            f"""
            Average:
                loss: {mean(losses)}
                next_sentence_loss: {mean(next_sentence_losses)}
                masked_language_loss: {mean(masked_language_losses)}
                next_sentence_acc: {mean(next_sentence_accs)}
                masked_language_acc: {mean(masked_language_accs)}
            """)
        return mean(losses)
    
    def save(self, path: Path, optimizer: torch.optim.Optimizer | None):
        path.mkdir(exist_ok=True)
        torch.save(
            {
                'model_state_dict': self.bert_pretrain.state_dict(),
                'pretrain_config': self.bert_pretrain.config,
                'bert_config': self.bert_pretrain.bert.config,
                'optimizer': optimizer
            },
            str(path / 'model.pt')
        )
        self.tokenizer.save(str(path / 'tokenizer.json'))
            
    def train(self, train_data: DataLoader, valid_data: DataLoader, epochs: int, save_dir: str, lr: float = 1e-4, valid_period: int = 10000):
        folder = Path(save_dir)
        regular_checkpoint_folder = folder / 'last'
        best_checkpoint_folder = folder / 'best'
        
        masked_language_loss_func = torch.nn.CrossEntropyLoss(ignore_index=0)
        next_sentence_loss_func = torch.nn.CrossEntropyLoss()
        
        best_loss = None
        
        total_steps = len(train_data) * epochs
        optimizer = torch.optim.Adam(self.bert_pretrain.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, total_steps=total_steps)
        reduce_on_plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        
        try:
            for epoch in range(epochs):
                print(f'Start training on {epoch + 1}.')
                self.train_on_epoch(
                    train_data, valid_data, masked_language_loss_func, next_sentence_loss_func, optimizer,
                    scheduler, reduce_on_plateau_scheduler, valid_period)
                loss = self.valid_on_epoch(valid_data, masked_language_loss_func, next_sentence_loss_func)
                
                if best_loss is None or best_loss > loss:
                    best_loss = loss
                    self.save(best_checkpoint_folder, optimizer)
        finally:
            self.save(regular_checkpoint_folder, optimizer)
