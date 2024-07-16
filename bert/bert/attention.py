import math
from torch import Tensor, nn


class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        assert dim % num_heads == 0, 'Dimension must be divisible by number of heads.'
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.queries_w = nn.Linear(dim, dim)
        self.keys_w = nn.Linear(dim, dim)
        self.values_w = nn.Linear(dim, dim)
        self.output_w = nn.Linear(dim, dim)
    
    def split_into_heads(self, x: Tensor) -> Tensor:
        """
            Splits input into heads heads.
            
            Args:
                x (torch.Tensor): tensor of shape (batch, seq_len, dim) into (batch, num_heads, seq_len, dim // num_heads)
        """
        
        batch, seq_len, dim = x.shape
        
        return x.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        
    def merge_from_heads(self, x: Tensor) -> Tensor:
        """
            Merges tensor with heads into one.
            
            Args:
                x (torch.Tensor): tensor of shape (batch, num_heads, seq_len, dim)
        """
        
        batch, num_heads, seq_len, head_dim = x.shape
        
        return x.transpose(1, 2).reshape(batch, seq_len, self.dim)
    
    @staticmethod
    def scaled_dot_attention(
        queries: Tensor, keys: Tensor, values: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """
            Computes scalar dot attention with attention mask.
            
            Args:
                queries (torch.Tensor): tensor of shape (batch, ..., queries_seq_len, dim)
                keys (torch.Tensor): tensor of shape (batch, ..., keys_seq_len, dim)
                values (torch.Tensor): tensor of shape (batch, ..., keys_seq_len, dim)
                attention_mask (torch.Tensor or None): tensor of shape (batch, ..., queries_seq_len, key_seq_len)
                
            Returns:
                torch.Tensor of shape (batch, ..., queries_seq_len, dim)
        """
        dim = queries.size(-1)
        
        attention_output = queries @ keys.transpose(-2, -1) / math.sqrt(dim)
        
        if attention_mask is not None:
            attention_output.masked_fill_(mask=~(attention_mask).bool(), value=float('-inf'))
            
        attention_scores = nn.functional.softmax(attention_output, dim=-1)
        
        return attention_scores @ values

    def forward(self, queries: Tensor, keys: Tensor, values: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """
            Multi head attention forward pass.
            
            Args:
                queries (torch.Tensor): tensor of shape (batch, queries_seq_len, dim)
                keys (torch.Tensor): tensor of shape (batch, keys_seq_len, dim)
                values (torch.Tensor): tensor of shape (batch, keys_seq_len, dim)
                attention_mask (torch.Tensor or None): tensor of shape (batch, num_heads, queries_seq_len, keys_seq_len)
                
            Returns:
                torch.Tensor of shape (batch, queries_seq_len, dim)
        """
        
        proj_queries = self.split_into_heads(self.queries_w(queries))
        proj_keys = self.split_into_heads(self.keys_w(keys))
        proj_values = self.split_into_heads(self.values_w(values))
        
        output = self.merge_from_heads(self.scaled_dot_attention(proj_queries, proj_keys, proj_values, attention_mask))
        return self.output_w(output)
