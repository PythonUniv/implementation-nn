import torch
from torch import Tensor, nn

from .multihead_attention import MultiHeadAttention
from .positional_encoding import PositionalEncoding
from .feed_forward import PositionWiseFeedForward


class DecoderBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        feed_forward_hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.self_mha = MultiHeadAttention(dim, dim, dim, num_heads)
        self.norm_self_mha = nn.LayerNorm(dim)
        self.cross_mha = MultiHeadAttention(dim, dim, dim, num_heads)
        self.norm_cross_mha = nn.LayerNorm(dim)
        self.feed_forward = PositionWiseFeedForward(dim, feed_forward_hidden_dim, dropout)
        self.norm_feed_forward = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        decoder_output: Tensor,
        encoder_output: Tensor,
        decoder_padding_mask: Tensor | None = None,
        encoder_padding_mask: Tensor | None = None,
        decoder_attention_mask: Tensor | None = None,
        encoder_attention_mask: Tensor | None = None
    ) -> Tensor:
        
        self_mha = self.self_mha(
            decoder_output, decoder_output, decoder_output,
            key_padding_mask=decoder_padding_mask, attention_mask=decoder_attention_mask)
        x = decoder_output + self.norm_self_mha(self_mha)
        
        cross_mha = self.cross_mha(
            x, encoder_output, encoder_output,
            key_padding_mask=encoder_padding_mask, attention_mask=encoder_attention_mask)
        x = x + self.norm_cross_mha(cross_mha)
        
        feed_forward = self.feed_forward(x)
        x = x + self.norm_feed_forward(feed_forward)
        
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        feed_forward_hidden_dim,
        num_heads: int, num_blocks: int,
        max_len: int = 10000,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embeddings = nn.Embedding(vocab_size, dim)
        self.positional_encoding = PositionalEncoding(dim, max_len)
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(dim, feed_forward_hidden_dim, num_heads, dropout) for idx in range(num_blocks)
        ])
    
    def forward(
        self,
        target: Tensor,
        encoder_output: Tensor,
        decoder_padding_mask: Tensor | None = None,
        encoder_padding_mask: Tensor | None = None,
        decoder_attention_mask: Tensor | None = None,
        encoder_attention_mask: Tensor | None = None
    ) -> Tensor:
        
        x = self.positional_encoding(self.embeddings(target))
        for decoder_block in self.decoder_blocks:
            x = decoder_block(
                decoder_output=x, encoder_output=encoder_output,
                decoder_padding_mask=decoder_padding_mask, encoder_padding_mask=encoder_padding_mask,
                decoder_attention_mask=decoder_attention_mask, encoder_attention_mask=encoder_attention_mask)
        return x
