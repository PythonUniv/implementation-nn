from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ViTConfig:
    image_size: int = 224
    patch_size: int = 16  # for image_size = 224 => 14 x 14 patches
    in_channels: int = 3
    dim: int = 768
    hidden_dim: int | None = None
    num_blocks: int = 12
    num_heads: int = 12
    mlp_activation: str = 'gelu'
    mlp_dropout = 0
    attention_dropout = 0
    attention_scale: float | None = None
    drop_path_attn_probs: list[float] | None = None
    drop_path_mlp_probs: list[float] | None = None
    num_classes: int | None = None
    attention_bias: bool = False
    
    
class PatchEmbeddings(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        
        self.config = config
        self.conv = nn.Conv2d(
            in_channels=config.in_channels, out_channels=config.dim, kernel_size=config.patch_size, stride=config.patch_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.conv(x)
        out = out.flatten(2).transpose(1, 2)
        return out


class PositionalEmbeddings(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        
        self.config = config
        self.num_patches = int((config.image_size / config.patch_size) ** 2 + 1)
        self.positional_embeddings = nn.Parameter(torch.zeros(1, self.num_patches, config.dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.positional_embeddings


class MLP(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        
        self.config = config
        hidden_dim = self.config.hidden_dim or 4 * config.dim
        self.fc_1 = nn.Linear(config.dim, hidden_dim)
        self.activation = getattr(torch.nn.functional, config.mlp_activation)
        self.fc_2 = nn.Linear(hidden_dim, config.dim)
        self.dropout = nn.Dropout(config.mlp_dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc_1(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc_2(out)
        return out


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        
        self.config = config
        self.queries_keys_values_weight = nn.Linear(config.dim, 3 * config.dim, bias=config.attention_bias)
        self.output_weight = nn.Linear(config.dim, config.dim)
        
        self.head_dim = config.dim // config.num_heads
        self.dropout = nn.Dropout(config.attention_dropout)
                        
    def scaled_dot_product_attention(
        self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, attention_mask: torch.Tensor | None = None,
        scale: float | None = None, dropout: nn.Dropout | None = None, return_scores: bool = False
    ) -> tuple[None | torch.Tensor] | tuple[torch.Tensor, torch.Tensor]:
        
        scale = scale or queries.size(-1) ** -0.5
        scores = queries @ keys.transpose(-1, -2) * scale
        if attention_mask is not None:
            scores.masked_fill_(~(attention_mask.bool()), value=float('-inf'))
        scores = F.softmax(scores, dim=-1)
        if dropout is not None:
            scores = dropout(scores)
        output = scores @ values
        if return_scores:
            return scores, output
        return None, output
    
    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None, return_scores: bool = False, flash_attention: bool = True
    ) -> tuple[None, torch.Tensor] | tuple[torch.Tensor, torch.Tensor]:
        if return_scores and flash_attention:
            raise AssertionError('Scores cannot be returned while flash_attention=True')
        
        batch_size, seq_len, dim = x.shape
        queries_keys_values: torch.Tensor = self.queries_keys_values_weight(x)
        queries_heads, keys_heads, values_heads = queries_keys_values.view(
            batch_size, seq_len, 3, self.config.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        
        scores = None
        if flash_attention:
            dropout = self.config.attention_dropout if self.training else 0
            out = F.scaled_dot_product_attention(
                queries_heads, keys_heads, values_heads, attention_mask, dropout, scale=self.config.attention_scale)
        else:
            scores, out = self.scaled_dot_product_attention(
                queries_heads, keys_heads, values_heads, attention_mask, scale=self.config.attention_scale,
                dropout=self.dropout, return_scores=return_scores)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
        out = self.output_weight(out)
        
        return scores, out
        

class DropPath(nn.Module):
    def __init__(self, prob: float):
        super().__init__()
        
        assert 0 <= prob <= 1, 'Probability should be in range 0 to 1.'
        self.prob = prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            batch_size = x.size(0)
            shape = (batch_size,) + (1,) * (x.dim() - 1)
            random = torch.floor(torch.rand(shape, device=x.device, dtype=x.dtype) + (1 - self.prob))
            return random * x
        else:
            return x
    

class Block(nn.Module):
    def __init__(self, config: ViTConfig, idx: int | None = None, flash_attention: bool = True):
        super().__init__()
        
        self.config = config
        self.flash_attention = flash_attention
        
        self.layer_norm_1 = nn.LayerNorm(config.dim)
        self.layer_norm_2 = nn.LayerNorm(config.dim)
        
        self.mlp = MLP(config)
        self.mha_attn = MultiHeadSelfAttention(config)

        drop_path_attn = config.drop_path_attn_probs[idx] if idx is not None and config.drop_path_attn_probs is not None else 0
        self.drop_path_attn = DropPath(drop_path_attn)
        
        drop_path_mlp = config.drop_path_mlp_probs[idx] if idx is not None and config.drop_path_mlp_probs is not None else 0
        self.drop_path_mlp = DropPath(drop_path_mlp)
    
    def forward(
        self, x: torch.Tensor, attention_mask: torch.Tensor | None = None, return_scores: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        out = self.layer_norm_1(x)
        scores, out = self.mha_attn(
            out, attention_mask=attention_mask, return_scores=return_scores, flash_attention=self.flash_attention)
        if return_scores:
            return scores, out
        out = x + self.drop_path_attn(out)
        out = out + self.drop_path_mlp(self.mlp(self.layer_norm_2(out)))
        return out
    

class VisionTransformer(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        
        self.config = config
        
        self.patch_embeddings = PatchEmbeddings(config)
        self.positional_embeddings = PositionalEmbeddings(config)
        self.blocks = nn.ModuleList([Block(config, idx, flash_attention=True) for idx in range(config.num_blocks)])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.dim))
        self.head = nn.Linear(config.dim, config.num_classes) if config.num_classes is not None else nn.Identity()
        self._flash_attention = True
        
        # init positional embeddings and class token weights
        nn.init.trunc_normal_(self.positional_embeddings.positional_embeddings, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
    
    def add_cls_token(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        out = torch.cat((self.cls_token.to(dtype=x.dtype, device=x.device).expand(batch_size, 1, -1), x), dim=1)
        return out
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.patch_embeddings(x)
        out = self.add_cls_token(out)
        out = self.positional_embeddings(out)
        for block in self.blocks:
            out = block(out)
        out = self.head(out[:, 0])
        return out
    
    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    
    @property
    def num_params(self) -> int:
        return sum(params.numel() for params in self.parameters())
    
    @property
    def flash_attention(self) -> bool:
        return self._flash_attention
    
    @flash_attention.setter
    def flash_attention(self, is_flash: bool):
        for block in self.blocks:
            block.flash_attention = is_flash
        self._flash_attention = is_flash
