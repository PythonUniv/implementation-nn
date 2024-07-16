import torch
from torch import Tensor, nn


class BertEmbedding(nn.Module):
    def __init__(self, vocab_size: int, dim: int, max_len: int, device: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.positional_embedding = nn.Embedding(max_len, dim)
        self.segment_embedding = nn.Embedding(2, dim)
        self.device = device
        
    def forward(self, x: Tensor, segment_mask: Tensor) -> Tensor:
        '''
            Return embedding from ids.
            
            Args:
                x (torch.Tensor): token ids with shape (batch, seq_len)
                segment_mask (torch.Tensor): tensor with segment ids (batch, seq_len)
                
            Returns:
                torch.Tensor of shape (batch_size, seq_len, embedding_dim).
        '''
        
        batch_size, seq_len = x.shape
        
        embedding = self.token_embedding(x) +\
                    self.positional_embedding(torch.arange(seq_len, device=self.device).repeat(repeats=(batch_size, 1))) +\
                    self.segment_embedding(segment_mask)
        return embedding
