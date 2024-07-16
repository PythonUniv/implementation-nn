import math
import torch
from torch import Tensor, nn

from .encoder import EncoderBlock
from .embedding import BertEmbedding


class Bert(nn.Module):
    def __init__(
        self, dim: int, hidden_dim: int, num_heads: int, num_blocks: int, vocab_size: int, device: str,
        dropout: float = 0.3, activation_function: str = 'gelu', max_len: int = 10_000,
        pad_idx: int = 0, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.activation_function = activation_function
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.embedding = BertEmbedding(vocab_size, dim, max_len, device)
        self.encoder_blocks = nn.ModuleList([EncoderBlock(dim, hidden_dim, num_heads, dropout, activation_function) for idx in range(num_blocks)])
        self.pad_idx = pad_idx
        self.device = device
        self.to(device)
    
    def forward(
        self, x: Tensor, segment_mask: Tensor | None = None, attention_mask: Tensor | None = None
    ) -> Tensor:
        """
            Bert forward pass.
            
            Args:
                x (torch.Tensor): tensor of shape (batch, seq_len)
                segment_mask (torch.Tensor or None): tensor of shape (batch, seq_len)
                attention_mask (torch.Tensor or None): tensor of shape (batch_seq_len)
                
            Returns:
                torch.Tensor of shape (batch, seq_len, dim)
        """
        
        if segment_mask is None:
            segment_mask = torch.zeros_like(x, dtype=torch.int, device=self.device)
        
        batch_size, seq_len = x.shape
        
        if attention_mask is None:
            attention_mask = (x != self.pad_idx).reshape(batch_size, 1, 1, seq_len)
        else:
            attention_mask = attention_mask.reshape(batch_size, 1, 1, seq_len)
        
        x = self.embedding(x, segment_mask) * math.sqrt(self.dim)
        
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, attention_mask)
            
        return x
    
    @property
    def config(self) -> dict:
        return {
            'dim': self.dim,
            'hidden_dim': self.hidden_dim,
            'num_heads': self.num_heads,
            'num_blocks': self.num_blocks,
            'activation_function': self.activation_function,
            'max_len': self.max_len,
            'vocab_size': self.vocab_size
        }
