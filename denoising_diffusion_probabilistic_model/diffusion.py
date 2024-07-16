from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DenoisingDiffusionConfig:
    num_groups: int = 32
    head_dim: int = 64
    in_channels: int = 3
    out_channels: int = 3
    channels: tuple[int, ...] = (64, 128, 256, 512, 512, 384)
    upsample: tuple[bool, ...] = (False, False, False, True, True, True)
    is_attention: tuple[bool, ...] = (False, True, False, False, False, True)
    time_steps: int = 1000
    conv_block_dropout: float = 0
    attention_bias: bool = True
    attention_dropout = 0
    attention_scale: float | None = None
    image_size: tuple[int, int] = (32, 32)


class SinusoidalPositionalEmbeddings(nn.Module):
    def __init__(self, config: DenoisingDiffusionConfig):
        super().__init__()
        
        self.config = config
        
        max_dim = max(config.channels)
        positions = torch.arange(0, config.time_steps, dtype=torch.float32).unsqueeze(1)
        terms = torch.pow(torch.arange(0, max_dim, 2) / max_dim, 10000)
        embeddings = torch.zeros(config.time_steps, max_dim)
        embeddings[:, ::2] = torch.sin(positions * terms)
        embeddings[:, 1::2] = torch.cos(positions * terms)
        embeddings.requires_grad_(False)
        self.register_buffer('embeddings', embeddings)
        
    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        batch_size, dim, h, w = x.shape
        return x + self.embeddings[time, :dim].view(batch_size, dim, 1, 1)


class ConvBlock(nn.Module):
    def __init__(self, config: DenoisingDiffusionConfig, idx: int):
        super().__init__()
        
        self.config = config
        channels = config.channels[idx]
        self.group_norm_1 = nn.GroupNorm(config.num_groups, channels)
        self.group_norm_2 = nn.GroupNorm(config.num_groups, channels)
        self.conv_1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(config.conv_block_dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.group_norm_1(x)
        out = self.conv_1(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.group_norm_2(out)
        out = self.conv_2(out)
        out = F.relu(out)
        return out
    

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config: DenoisingDiffusionConfig, idx: int, flash_attention: bool = True):
        super().__init__()
        
        self.config = config
        self.flash_attention = flash_attention
        dim = config.channels[idx]
        self.head_dim = config.head_dim
        self.num_heads = dim // self.head_dim
        self.queries_keys_values_weights = nn.Linear(dim, 3 * dim, config.attention_bias)
        self.output_weights = nn.Linear(dim, dim)
    
    def scaled_dot_product_attention(
        self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
        attn_mask: torch.Tensor | None = None, scale: float | None = None, dropout: float = 0
    ) -> torch.Tensor:
        scale = scale or queries.size(-1) ** -0.5
        scores = queries @ keys.transpose(1, 2) * scale
        if dropout:
            scores = F.dropout(scores, dropout)
        if attn_mask is not None:
            scores.masked_fill_(~attn_mask.bool(), value=float('-inf'))
        scores = scores.softmax(dim=-1)
        out_values = scores @ values
        return out_values
    
    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        queries_keys_values = self.queries_keys_values_weights(x)
        num_heads = dim // self.config.head_dim
        queries_head, keys_head, values_head = queries_keys_values.view(
            batch_size, seq_len, 3, num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        
        dropout = self.config.attention_dropout if self.training else 0
        if self.flash_attention:
            out = F.scaled_dot_product_attention(
                queries_head, keys_head, values_head, attn_mask, dropout, scale=self.config.attention_scale)
        else:
            out = self.scaled_dot_product_attention(
                queries_head, keys_head, values_head, attn_mask, self.config.attention_scale, dropout)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
        out = self.output_weights(out)
        return out


class Block(nn.Module):
    def __init__(self, config: DenoisingDiffusionConfig, embeddings: SinusoidalPositionalEmbeddings, idx: int):
        super().__init__()
        
        self.config = config
        self.is_attention = config.is_attention[idx]
        self.upscale = config.upsample[idx]
    
        self.conv_block_1 = ConvBlock(config, idx)
        self.conv_block_2 = ConvBlock(config, idx)
        self.embeddings = embeddings
        if config.is_attention[idx]:
            self.attention = MultiHeadSelfAttention(config, idx)
        channels = config.channels[idx]
        if config.upsample[idx]:
            self.conv = nn.ConvTranspose2d(channels, channels // 2, kernel_size=4, stride=2, padding=1)
        else:
            self.conv = nn.Conv2d(channels, 2 * channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor, time: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.embeddings(x, time)
        out = out + self.conv_block_1(out)
        if self.is_attention:
            batch_size, channels, h, w = out.shape
            out = out.permute(0, 2, 3, 1).view(batch_size, h * w, channels)
            out = self.attention(out)
            out = out.transpose(1, 2).contiguous().view(batch_size, channels, h, w)
        out = self.embeddings(out, time)
        out = out + self.conv_block_2(out)
        saved = out
        out = self.conv(out)
        return saved, out
    

class DiffusionModel(nn.Module):
    def __init__(self, config: DenoisingDiffusionConfig):
        super().__init__()
        
        self.config = config
        
        self.conv_1 = nn.Conv2d(config.in_channels, config.channels[0], kernel_size=3, padding=1)
        out_channels_blocks = config.channels[0] + config.channels[-1] // 2 
        self.conv_2 = nn.Conv2d(out_channels_blocks, out_channels_blocks // 2, kernel_size=3, padding=1)
        self.conv_3 = nn.Conv2d(out_channels_blocks // 2, self.config.out_channels, kernel_size=1)
        embeddings = SinusoidalPositionalEmbeddings(config)
        self.blocks = nn.ModuleList([Block(config, embeddings, idx) for idx in range(len(config.channels))])
    
    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        saved = []
        
        out = self.conv_1(x)
        for block in self.blocks:
            if block.upscale:
                out = block(out, time)[1]
                save = saved.pop()
                out = torch.cat((save, out), dim=1)
            else:
                save, out = block(out, time)
                saved.append(save)
        out = self.conv_2(out)
        out = F.relu(out)
        out = self.conv_3(out)
        return out
