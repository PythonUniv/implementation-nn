import torch
from torch import Tensor, nn


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int, max_len: int = 10000):
        super().__init__()
             
        pe = torch.zeros(max_len, embedding_dim)
        
        positions = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        indices = torch.pow(torch.arange(0, embedding_dim, 2, dtype=torch.float), exponent=1e-4)
        
        pe[:, ::2] = torch.sin(positions * indices)
        pe[:, 1::2] = torch.cos(positions * indices)
        
        pe.unsqueeze_(0)
        
        self.register_buffer(name='pe', tensor=pe)
    
    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[:, : x.size(1)]
