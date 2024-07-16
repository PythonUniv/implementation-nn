import math
from torch import Tensor, nn

from .multihead_attention import MultiHeadAttention
from .positional_encoding import PositionalEncoding
from .feed_forward import PositionWiseFeedForward


class EncoderBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        feed_forward_hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.mha = MultiHeadAttention(dim, dim, dim, num_heads)
        self.feed_forward = PositionWiseFeedForward(dim, feed_forward_hidden_dim, dropout)
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: Tensor, src_padding_mask: Tensor | None = None) -> Tensor:
        attention_output = self.mha(x, x, x, key_padding_mask=src_padding_mask)
        x = x + self.dropout(self.norm_1(attention_output))
        x = x + self.norm_2(self.feed_forward(x))
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        feed_forward_hidden_dim: int,
        num_heads: int,
        num_blocks: int,
        max_len: int = 10000,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.dim = dim
        
        self.embeddings = nn.Embedding(vocab_size, dim)
        self.positional_encoding = PositionalEncoding(dim, max_len=max_len)
        self.encoder_blocks = nn.ModuleList(
            [EncoderBlock(dim, feed_forward_hidden_dim, num_heads, dropout) for idx in range(num_blocks)])
        
    def forward(self, x: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        embeddings = math.sqrt(self.dim) * self.embeddings(x)
        x = self.positional_encoding(embeddings)
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x=x, src_padding_mask=padding_mask)
        return x
