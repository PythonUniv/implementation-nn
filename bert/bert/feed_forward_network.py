import torch
from torch import Tensor, nn


class FeedForwardNetwork(nn.Module):
    def __init__(
        self, dim: int, hidden_dim: int,
        dropout: float = 0.3, activation_func: str = 'gelu',
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        self.linear_1 = nn.Linear(dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, dim)
        self.dropout_layer = nn.Dropout(dropout)
        self.activation_func = getattr(torch.nn.functional, activation_func)
        
    def forward(self, x: Tensor) -> Tensor:
        """
            Fully connected feed forward with hidden layer and normalization layer of output.
        
            Args:
                x (torch.Tensor): tensor of shape (batch_idx, ..., dim)
                
            Returns:
                torch.Tensor of the same shape as shape of input tensor.
        """
        
        x = self.activation_func(self.linear_1(x))
        x = self.linear_2(self.dropout_layer(x))
        return x
