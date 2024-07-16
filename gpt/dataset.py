import os
import numpy as np
import torch
import torch.nn.functional as F


class DataLoaderLite:
    def __init__(self, data_root: str, batch_size: int, seq_len: int, process_rank: int, num_processes: int, split: str):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.process_rank = process_rank
        self.num_processes = num_processes
        
        assert split in {'train', 'test'}
        
        self.shards = sorted(
            [os.path.join(data_root, file_name) for file_name in os.listdir(data_root) if split in file_name])
        
        assert self.shards, 'Any shard is not found.'
        self.reset()
        
    @staticmethod
    def load_tokens(path: str) -> torch.Tensor:
        return torch.tensor(np.load(path).astype(np.int32), dtype=torch.long)
    
    def reset(self):
        self.current_shard_idx = 0
        
        shard = self.shards[0]
        self.tokens = self.load_tokens(shard)
        self.current_position = self.batch_size * self.seq_len * self.process_rank
        
    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        buffer = self.tokens[self.current_position: self.current_position + self.batch_size * self.seq_len + 1]
        x = buffer[: -1].view(self.batch_size, self.seq_len)
        y = buffer[1:].view(self.batch_size, self.seq_len)
        
        self.current_position += self.batch_size * self.seq_len * self.num_processes
        
        if self.current_position + (self.batch_size * self.seq_len * self.num_processes + 1) < len(self.tokens):
            self.current_shard_idx = (self.current_shard_idx + 1) % len(self.shards)
            shard = self.shards[self.current_shard_idx]
            self.tokens = self.load_tokens(shard)
            self.current_position = self.batch_size * self.seq_len * self.process_rank
        
        return x, y


def get_most_likely_row(tokens: torch.Tensor, mask: torch.Tensor, logits: torch.Tensor) -> float:
    shift_logits = logits[..., : -1, :].contiguous()
    shift_tokens = tokens[..., 1:].contiguous()
    
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    
    losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none').view(tokens.size(0), -1)
    
    shift_mask = mask[..., 1:].contiguous()
    masked_shift_losses = shift_mask * losses
    average_loss = masked_shift_losses.sum(dim=1) / shift_mask.sum(dim=1) 
    
    pred_norm = average_loss.argmin().item()
    return pred_norm
