from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import StochasticDepth


@dataclass
class SegFormerConfig:
    in_channels: int = 3
    stage_channels_out: tuple[int] = (48, 96, 160)
    stage_mlp_hidden: tuple[int] = (4 * 48, 4 * 96, 4 * 160)
    stage_attn_head_dim: tuple[int] = (16, 16, 16)
    attn_reduction: tuple[int] = (4, 4, 4)
    overlap_patch_sizes: tuple[int] = (3, 3, 3)
    overlap_stride_sizes: tuple[int] = (4, 2, 2)
    stage_num_blocks: tuple[int] = (4, 6, 2)
    stage_drop_paths: tuple[tuple[float]] = ((0, 0.05, 0.1, 0.15), (0.2, 0.25, 0.3, 0.35, 0.4, 0.45), (0.5, 0.55))
    decoder_out_channels: int = 128
    num_classes: int = 100
    
    
class LayerNorm2d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.layer_norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


class SegFormerAttention(nn.Module):
    def __init__(
        self, channels: int, head_dim: int, reduction: int = 4,
        qkv_bias: bool = False, qk_scale: float | None = None
    ):
        super().__init__()
        
        self.channels = channels
        self.head_dim = head_dim
        self.num_heads = channels // head_dim
        self.query_linear = nn.Linear(channels, channels, bias=qkv_bias)
        self.conv = nn.Conv2d(channels, channels, kernel_size=reduction, stride=reduction, bias=False)
        self.layer_norm_2d = LayerNorm2d(channels)
        self.key_value_linear = nn.Linear(channels, 2 * channels, bias=qkv_bias)
        self.output_linear = nn.Linear(channels, channels)
        self.qk_scale = qk_scale or head_dim ** -0.5
        self.flash = True
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        
        queries = x.flatten(2).transpose(1, 2)
        queries = self.query_linear(queries)
        queries_heads = queries.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
                
        reduction_x = self.conv(x)
        reduction_x = self.layer_norm_2d(reduction_x)
        reduction_x = reduction_x.flatten(2).transpose(1, 2)
        reduction_x = self.key_value_linear(reduction_x)
        keys_values_heads = reduction_x.view(b, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        keys_heads, values_heads = keys_values_heads[0], keys_values_heads[1]
                
        x = self.scalar_dot_product_attention(queries_heads, keys_heads, values_heads)
        x = x.transpose(1, 2).contiguous().view(b, h * w, c)
        x = self.output_linear(x)
        x = x.view(b, h, w, c).permute(0, 3, 1, 2)
        return x
        
    def scalar_dot_product_attention(
        self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor
    ) -> torch.Tensor:
        if self.flash:
            attention = F.scaled_dot_product_attention(queries, keys, values, scale=self.qk_scale)
        else:
            attention_scores = queries @ keys.transpose(-2, -1) * self.qk_scale
            attention_scores = attention_scores.softmax(dim=-1)
            attention = attention_scores @ values
        return attention
    
    
class MixMLP(nn.Module):
    def __init__(self, channels: int, hidden: int):
        super().__init__()
        self.conv_1 = nn.Conv2d(channels, channels, 1)
        self.conv_2 = nn.Conv2d(channels, hidden, kernel_size=3, padding=1, groups=channels)
        self.gelu = nn.GELU()
        self.conv_3 = nn.Conv2d(hidden, channels, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.gelu(x)
        x = self.conv_3(x)
        return x
    

class OverlapPatchEmbedding(nn.Module):
    def __init__(self, config: SegFormerConfig, stage: int):
        super().__init__()
        
        in_channels = config.in_channels if stage == 0 else config.stage_channels_out[stage - 1]
        out_channels = config.stage_channels_out[stage]
        kernel_size = config.overlap_patch_sizes[stage]
        stride = config.overlap_stride_sizes[stage]
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=kernel_size // 2, bias=False)
        self.layer_norm_2d = LayerNorm2d(out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.layer_norm_2d(x)
        return x
    

class Block(nn.Module):
    def __init__(self, config: SegFormerConfig, stage: int, idx: int):
        super().__init__()
        
        channels = config.stage_channels_out[stage]
        head_dim = config.stage_attn_head_dim[stage]
        reduction = config.attn_reduction[stage]
        hidden = config.stage_mlp_hidden[stage]
        drop_path = config.stage_drop_paths[stage][idx]
        
        self.segformer_attn = SegFormerAttention(channels, head_dim, reduction)
        self.mix_mlp = MixMLP(channels, hidden)
        self.layer_norm_2d_1 = LayerNorm2d(channels)
        self.layer_norm_2d_2 = LayerNorm2d(channels)
        self.drop_path = StochasticDepth(drop_path, 'batch')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.segformer_attn(self.layer_norm_2d_1(x))
        x = x + self.drop_path(self.mix_mlp(self.layer_norm_2d_2(x)))
        return x
    

class EncoderStage(nn.Module):
    def __init__(self, config: SegFormerConfig, stage: int):
        super().__init__()
        
        num_blocks = config.stage_num_blocks[stage]
        out_channels = config.stage_channels_out[stage]
        self.overlap_patch_embedding = OverlapPatchEmbedding(config, stage)
        self.blocks = nn.ModuleList(Block(config, stage, idx) for idx in range(num_blocks))
        self.layer_norm_2d = LayerNorm2d(out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.overlap_patch_embedding(x)
        for block in self.blocks:
            x = block(x)
        x = self.layer_norm_2d(x)
        return x


class SegFormerEncoder(nn.Module):
    def __init__(self, config: SegFormerConfig):
        super().__init__()
        
        num_stages = len(config.stage_channels_out)
        self.stages = nn.ModuleList(EncoderStage(config, idx) for idx in range(num_stages))
    
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        stages_out = []
        for stage in self.stages:
            x = stage(x)
            stages_out.append(x)
        return stages_out
    
    
class StageDecoder(nn.Module):
    def __init__(self, config: SegFormerConfig, stage: int):
        super().__init__()
        
        stage_out_channels = config.stage_channels_out[stage]
        decoder_out_channels = config.decoder_out_channels
        
        scale = (torch.cumprod(torch.tensor(config.overlap_stride_sizes), dim=0) // 4)[stage].item()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=scale)
        self.conv = nn.Conv2d(stage_out_channels, decoder_out_channels, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.conv(x)
        return x
    
    
class SegFormerDecoder(nn.Module):
    def __init__(self, config: SegFormerConfig):
        super().__init__()
        
        num_stages = len(config.stage_channels_out)
        self.decoder_stages = nn.ModuleList(StageDecoder(config, stage) for stage in range(num_stages))
        
    def forward(self, stages_out: list[torch.Tensor]) -> torch.Tensor:
        decoded_stages = []
        for stage_out, decoder_stage in zip(stages_out, self.decoder_stages):
            decoded = decoder_stage(stage_out)
            decoded_stages.append(decoded)
        x = torch.cat(decoded_stages, dim=1)
        return x
    
    
class SegFormerHead(nn.Module):
    def __init__(self, config: SegFormerConfig):
        super().__init__()
        
        in_channels = len(config.stage_channels_out) * config.decoder_out_channels
        hidden_channels = config.decoder_out_channels
        self.conv_1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        self.batch_norm_2d = nn.BatchNorm2d(hidden_channels)
        self.conv_2 = nn.Conv2d(hidden_channels, config.num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_1(x)
        x = self.relu(x)
        x = self.batch_norm_2d(x)
        x = self.conv_2(x)
        return x


class SegFormer(nn.Module):
    def __init__(self, config: SegFormerConfig):
        super().__init__()
        
        self.config = config
        self.segformer_encoder = SegFormerEncoder(config)
        self.segformer_decoder = SegFormerDecoder(config)
        self.segformer_head = SegFormerHead(config)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stages_out = self.segformer_encoder(x)
        x = self.segformer_decoder(stages_out)
        x = self.segformer_head(x)
        return x


if __name__ == '__main__':
    config = SegFormerConfig()

    segformer = SegFormer(config).to('cuda')
    print(f'Parameters: {sum(params.numel() for params in segformer.parameters()):,}')
    
    x = torch.randn(16, 3, 128, 128).to('cuda')
    x = segformer(x)
