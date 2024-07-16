from torch import Tensor, nn

from .attention import MultiHeadAttention
from .feed_forward_network import FeedForwardNetwork


class EncoderBlock(nn.Module):
    def __init__(
        self, dim: int, hidden_dim: int, num_heads: int,
        dropout: float = 0.3, activation_function: str = 'gelu', *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        
        self.feed_forward = FeedForwardNetwork(dim, hidden_dim, dropout, activation_function)
        self.mha = MultiHeadAttention(dim, num_heads)
        self.layer_norm_1 = nn.LayerNorm(dim)
        self.layer_norm_2 = nn.LayerNorm(dim)
        
    def forward(self, x: Tensor, attention_mask: Tensor | None) -> Tensor:
        """
            Encoder forward pass.
            
            Args:
                x (torch.Tensor): tensor of shape (batch, seq_len, dim)
                attention_mask (torch.Tensor or None): tensor of shape (batch, num_heads, seq_len, seq_len)
                
                
            Implemented effective layer normalization regarding paper: https://proceedings.mlr.press/v119/xiong20b/xiong20b.pdf
        """
        
        x_normed = self.layer_norm_1(x)
        x = x + self.mha(x_normed, x_normed, x_normed, attention_mask)
        x_normed = self.layer_norm_2(x)
        x = x + self.feed_forward(x_normed)
        
        return x
