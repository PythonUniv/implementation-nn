import torch
from torch import Tensor, nn

from .bert import Bert
from .utils import flatten_2d


class BertPretrain(nn.Module):
    def __init__(
        self, bert: Bert, ml_hidden: int, ns_hidden: int, device: str, dropout: float = 0.3, activation_function: str = 'gelu',
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.bert = bert
        
        self.ml_hidden = ml_hidden
        self.ns_hidden = ns_hidden
        self.activation_function = activation_function
        
        self.masked_language_linear_1 = nn.Linear(bert.dim, ml_hidden)
        self.masked_language_linear_2 = nn.Linear(ml_hidden, bert.vocab_size)
        self.next_sentence_linear_1 = nn.Linear(bert.dim, ns_hidden)
        self.next_sentence_linear_2 = nn.Linear(ns_hidden, 2)
        self.dropout = nn.Dropout(dropout)
        self.activation_func = getattr(nn.functional, activation_function)
        self.device = device
        
    def forward(
        self, x: Tensor, masked_positions: Tensor | list[list[int]], segment_mask: Tensor | None = None, 
        attention_mask: Tensor | None = None
    ) -> tuple[Tensor]:
        """
            Pre training forward pass.
            
            Args:
                x (torch.Tensor): tensor of shape (batch, seq_len)
                segment_mask (torch.Tensor or None): tensor of shape (batch, seq_len)
                attention_mask (torch.Tensor or None): tensor of shape (batch, seq_len)
                
            Returns:
                tuple[Tensor]: (torch.Tensor of shape (batch * masked_language_tokens, vocab_size), torch.Tensor of shape (batch, vocab_size))
        """
        
        bert_forward = self.bert(x, segment_mask, attention_mask)
        
        positions = [idx for idx, mask in enumerate(masked_positions) for item in mask]
        
        masked_language = self.dropout(self.activation_func(
            self.masked_language_linear_1(bert_forward[positions, flatten_2d(masked_positions)])))
        masked_language = self.masked_language_linear_2(masked_language)

        next_sentence = self.dropout(self.activation_func(self.next_sentence_linear_1(bert_forward[:, 0, :])))
        next_sentence = self.next_sentence_linear_2(next_sentence)
        
        return masked_language, next_sentence
    
    @property
    def config(self) -> dict:
        return {
            'ml_hidden': self.ml_hidden,
            'ns_hidden': self.ns_hidden,
            'activation_function': self.activation_function
        }
