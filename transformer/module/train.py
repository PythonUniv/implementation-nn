from pathlib import Path
import numpy as np
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard.writer import SummaryWriter
from typing import Callable
import tqdm
import wandb

from .transformer import Transformer
from .dataset import TranslationDataset
from .tokenizer import Tokenizer


pad_token_idx = 0


wandb_step = 0


def train_epoch(
    model: Transformer,
    data_loader: DataLoader,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    summary_writer: SummaryWriter,
    epoch_idx: int,
    wandb_logs_period: int | None = None
) -> list[float]:
    
    global wandb_step
    
    model.train(True)
    
    losses = []
    
    with tqdm.tqdm(data_loader, position=0, leave=True) as data:
        for batch_idx, (x, y) in enumerate(data):
            x, y = x.to(model.device), y.to(model.device)
            
            optimizer.zero_grad()
            
            outputs = model(x, y[:, :-1])
            loss = loss_fn(outputs.contiguous().view(-1, model.vocab_size), y[:, 1:].contiguous().view(-1))
            losses.append(loss.item())
            data.set_description(f'Train model: {epoch_idx + 1} epoch. Loss: {loss.item():.3f}')
            
            loss.backward()
            optimizer.step()
            
            global_step = len(data_loader) * epoch_idx + batch_idx
            summary_writer.add_scalar('train/loss', loss.item(), global_step)
            summary_writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)
            
            if wandb_logs_period is not None:
                commit = (batch_idx + 1) % wandb_logs_period == 0
                wandb.log({'train/loss': loss.item(), 'train/lr': optimizer.param_groups[0]['lr']},
                          step=wandb_step, commit=commit)
            wandb_step += 1
            
            scheduler.step()
            
    return losses


def evaluate(
    model: Transformer,
    data_loader: DataLoader,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    pad_token_idx: int,
    summary_writer: SummaryWriter,
    epoch_idx: int,
    wandb_logs_period: int | None = None
) -> tuple[list[float], list[float]]:
    
    global wandb_step
    
    model.eval()
    
    losses, accuracies = [], []
    with tqdm.tqdm(data_loader, position=0, leave=True) as data:
        data.set_description(f'Evaluation: {epoch_idx + 1} epoch.')
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(data):
                x, y = x.to(model.device), y.to(model.device)
                outputs: Tensor = model(x, y[:, :-1])
                
                loss = loss_fn(outputs.contiguous().view(-1, model.vocab_size),
                            y[:, 1:].contiguous().view(-1))
                losses.append(loss.item())
                
                predictions = outputs.argmax(dim=-1)
                
                predictions_truth_padding = predictions.masked_fill(y[:, 1:] == pad_token_idx, pad_token_idx)
                accuracy = (y[:, 1:] == predictions_truth_padding).float().mean()
                
                accuracies.append(accuracy.item())
                
                global_step = len(data_loader) * epoch_idx + batch_idx
                summary_writer.add_scalar('validation/loss', loss.item(), global_step)
                summary_writer.add_scalar('validation/accuracy', accuracy.item(), global_step)
                
                if wandb_logs_period is not None:
                    commit = (batch_idx + 1) % wandb_logs_period == 0
                    wandb.log({'validation/loss': loss.item(), 'validation/accuracy': accuracy},
                              commit=commit, step=wandb_step)
                wandb_step += 1
            
        return losses, accuracies
    
    
def collate_fn(batch) -> tuple[Tensor, Tensor]:
    return (pad_sequence([pair[0] for pair in batch], batch_first=True, padding_value=pad_token_idx),
            pad_sequence([pair[1] for pair in batch], batch_first=True, padding_value=pad_token_idx))


def train(
    model: Transformer,
    source_sentences: list[str],
    target_sentences: list[str],
    epochs: int,
    batch_size: int,
    save_folder: str,
    max_lr: float = 1e-3,
    train_ratio: float = 0.8,
    vocab_size: int = 10000,
    pad_token: str = '[PAD]',
    sos_token: str = '[SOS]',
    eos_token: str = '[EOS]',
    unk_token: str = '[UNK]',
    log_dir: str | None = None,
    wandb_logs_period: int | None = None,
    wandb_api_key: str | None = None
):
    
    num_model_params = sum([torch.prod(torch.tensor(parameter.size())).item() for parameter in model.parameters()])
    print(f'Training model with {num_model_params} parameters.')
    
    print(f'Source sentences: {len(source_sentences)}')
    print(f'Target sentences: {len(target_sentences)}')
    
    model.init_weights()
    
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_token_idx, label_smoothing=0.05)
    optimizer = torch.optim.Adam(model.parameters(), lr=max_lr)
    
    steps_per_epoch = len(source_sentences) // batch_size
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=max_lr, steps_per_epoch=steps_per_epoch, epochs=epochs, anneal_strategy='linear')
    summary_writer = SummaryWriter(log_dir)

    source_tokenizer = Tokenizer(vocab_size, pad_token, sos_token, eos_token, unk_token)
    target_tokenizer = Tokenizer(vocab_size, pad_token, sos_token, eos_token, unk_token)
    
    source_tokenizer.train(source_sentences)
    target_tokenizer.train(target_sentences)
    
    train_size = int(train_ratio * len(source_sentences))
    dataset = TranslationDataset(source_sentences, target_sentences, source_tokenizer, target_tokenizer)
    train_dataset = Subset(dataset, indices=range(train_size))
    val_dataset = Subset(dataset, indices=range(train_size, len(source_sentences)))
    
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    best_model_path = Path(save_folder) / 'best_model.pt'
    best_model_state_dict_path = Path(save_folder) / 'best_model_state_dict.pt'
    
    if wandb_logs_period is not None:
        wandb.login(key=wandb_api_key)
        wandb.init(
            project='transformer',
            config={
                    'dim': model.dim,
                    'feed_forward_dim': model.feed_forward_hidden_dim,
                    'num_encoder_blocks': model.num_encoder_blocks,
                    'num_decoder_blocks': model.num_decoder_blocks,
                    'dataset_size': len(source_sentences),
                    'batch_size': batch_size,
                    'lr': max_lr
            }
        )
        
    try:
        best_accuracy = None
        
        for epoch in range(epochs):
            train_epoch(model, train_data_loader, loss_fn, optimizer, scheduler, summary_writer, epoch, wandb_logs_period)
            losses, accuracies = evaluate(model, val_data_loader, loss_fn, pad_token_idx, summary_writer, epoch, wandb_logs_period)
            print(f'Average loss for epoch: {np.mean(losses)}, Average accuracy for epoch: {np.mean(accuracies)}')
            
            accuracy = np.mean(accuracies)
            if best_accuracy is None or best_accuracy < accuracy:
                best_accuracy = accuracy
                torch.save(model, best_model_path)
                torch.save(model.get_state_dict(), best_model_state_dict_path)
                
    finally:
        model_path = Path(save_folder) / 'model.pt'
        model_state_dict_path = Path(save_folder) / 'model_state_dict.pt'
        state_dict = model.get_state_dict()
        
        torch.save(state_dict, model_state_dict_path)
        torch.save(model, model_path)
        
        source_tokenizer_path = Path(save_folder) / 'source_tokenizer.tok'
        target_tokenizer_path = Path(save_folder) / 'target_tokenizer.tok'
        source_tokenizer.save(source_tokenizer_path)
        target_tokenizer.save(target_tokenizer_path)
        
        if wandb_api_key:
            wandb.save(model_state_dict_path)
            wandb.save(model_path)
            wandb.save(best_model_state_dict_path)
            wandb.save(best_model_path)
            wandb.save(source_tokenizer_path)
            wandb.save(target_tokenizer_path)
