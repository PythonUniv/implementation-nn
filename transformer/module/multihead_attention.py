import math
import torch
from torch import nn
from torch import Tensor


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_heads: int):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        
        self.proj_queries = nn.Linear(input_dim, hidden_dim)
        self.proj_keys = nn.Linear(input_dim, hidden_dim)
        self.proj_values = nn.Linear(input_dim, hidden_dim)
        self.proj_output = nn.Linear(hidden_dim, output_dim)
    
    @staticmethod
    def scaled_dot_attention(
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        attention_mask: Tensor | None = None,
        key_padding_mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """
            queries: tensor of shape (batch_size, num_heads, queries_seq_len, queries_dim)
            keys: tensor of shape (batch_size, num_heads, keys_seq_len, queries_dim)
            values: tensor of shape (batch_size, num_heads, keys_seq_len, values_dim)
            attention_mask: tensor of shape (queries_seq_len, keys_seq_len) or None
            
            Returns: tuple of attention - tensor of shape (batch_size, num_heads, queries_seq_len, keys_seq_len)
            and values - tensor of shape (batch_size, num_heads, queries_seq_len, values_dim) 
        """
        
        keys_dim = keys.size(-1)
        
        logits = queries @ keys.transpose(-1, -2) / math.sqrt(keys_dim)
        
        if attention_mask is not None:
            logits += attention_mask
            
            if key_padding_mask is not None:
                logits += key_padding_mask.unsqueeze(1).unsqueeze(2)
        
        attention_scores = torch.softmax(logits, dim=-1)
        return attention_scores, attention_scores @ values

    def split_into_heads(self, tensor: Tensor) -> Tensor:
        """
            Convert tensor of shape (batch_size, seq_len, tensor_dim) into
            tensor of shape (batch_size, num_heads, seq_len, tensor_dim // num_heads)
        """
        
        batch_size, seq_len, tensor_dim = tensor.size()
        heads_dim = tensor_dim // self.num_heads
        
        return tensor.view(batch_size, seq_len, self.num_heads, heads_dim).transpose(1, 2)
    
    def combine_heads(self, tensor: Tensor) -> Tensor:
        """
            Convert tensor of shape (batch_size, num_heads, seq_len, heads_dim) into
            tensor of shape (batch_size, seq_len, num_heads * heads_dim)
        """
        
        batch_size, num_heads, seq_len, heads_dim = tensor.size()
        return tensor.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
    
    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        attention_mask: Tensor | None = None,
        key_padding_mask: Tensor | None = None
    ) -> Tensor:
        """MultiHeadAttention forward pass."""
        
        proj_queries = self.proj_queries(queries)
        proj_keys = self.proj_keys(keys)
        proj_values = self.proj_values(values)

        multi_head_queries = self.split_into_heads(proj_queries)
        multi_head_keys = self.split_into_heads(proj_keys)
        multi_head_values = self.split_into_heads(proj_values)
        
        attention_scores, values = self.scaled_dot_attention(
            queries=multi_head_queries,
            keys=multi_head_keys,
            values=multi_head_values,
            attention_mask=attention_mask,
            key_padding_mask=key_padding_mask
        )
        
        outputs = self.proj_output(self.combine_heads(values))
        
        return outputs
