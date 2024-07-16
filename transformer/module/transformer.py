import torch
from torch import Tensor, nn

from .encoder import Encoder
from .decoder import Decoder


class Transformer(nn.Module):
    def __init__(
        self,
        dim: int,
        vocab_size: int,
        num_heads: int,
        num_encoder_blocks: int,
        num_decoder_blocks: int,
        feed_forward_hidden_dim: int,
        sos_token_idx: int,
        eos_token_idx: int,
        pad_token_idx: int,
        max_len: int = 10000,
        dropout: float = 0.1,
        device: str = 'cuda'
    ):
        super().__init__()
        
        self.encoder = Encoder(
            vocab_size=vocab_size,
            dim=dim,
            feed_forward_hidden_dim=feed_forward_hidden_dim,
            num_heads=num_heads,
            num_blocks=num_encoder_blocks,
            max_len=max_len,
            dropout=dropout)
        
        self.decoder = Decoder(
            vocab_size=vocab_size,
            dim=dim,
            feed_forward_hidden_dim=feed_forward_hidden_dim,
            num_heads=num_heads,
            num_blocks=num_decoder_blocks,
            max_len=max_len,
            dropout=dropout)
        
        self.linear = nn.Linear(dim, vocab_size)
        
        self.dim = dim
        self.num_heads = num_heads
        self.num_encoder_blocks = num_encoder_blocks
        self.num_decoder_blocks = num_decoder_blocks
        self.feed_forward_hidden_dim = feed_forward_hidden_dim
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.sos_token_idx = sos_token_idx
        self.eos_token_idx = eos_token_idx
        self.pad_token_idx = pad_token_idx
        self.dropout = dropout
        
        self.stop_generation_tokens = {self.eos_token_idx, self.pad_token_idx}
        
        self.device = device
        
    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
            x: Tensor - tensor of shape (batch_size, seq_len)
            Returns: - tuple(tensor of shape (batch_size, padding_mask), tensor of shape (batch_size, seq_len, dim))
        """
        
        # attention mask taking into account padding
        padding_mask = self.padding_mask(x)
        
        encoder_output = self.encoder(x=x, padding_mask=padding_mask)
        return padding_mask, encoder_output
    
    def decode(
        self,
        encoder_output: Tensor,
        target: Tensor,
        encoder_padding_mask: Tensor | None = None
    ) -> Tensor:
        
        decoder_output = self.decoder(
            target=target,
            encoder_output=encoder_output,
            decoder_padding_mask=self.padding_mask(target),
            encoder_padding_mask=encoder_padding_mask,
            decoder_attention_mask=self.decoder_attention_mask(target))
        return self.linear(decoder_output)
    
    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        encoder_padding_mask, encoder_output = self.encode(x)
        outputs = self.decode(encoder_output, target, encoder_padding_mask=encoder_padding_mask)
        return outputs
        
    def padding_mask(self, x: Tensor, value: float | None = None) -> Tensor:
        batch_size, seq_len = x.size(0), x.size(1)
        padding_mask = torch.zeros(batch_size, seq_len, device=self.device).float()
        return padding_mask.masked_fill(
            x == self.pad_token_idx, float('-inf') if value is None else float(value)
        )
    
    def decoder_attention_mask(self, x: Tensor, value: float | None = None) -> Tensor:
        seq_len = x.size(1)
        return torch.where(torch.tril(torch.ones(seq_len, seq_len)).bool(),
                           0, float('-inf') if value is None else value).to(self.device)
        
    def generate(self, input: Tensor, max_len: int = 1000) -> Tensor:
        x = input.unsqueeze(0)
        
        encoder_padding_mask, encoder_output = self.encode(x)
        
        outputs = self.sos_token_idx * torch.ones(1, max_len).type_as(x).to(self.device)
        
        for step in range(1, max_len):
            target = outputs[:, :step]
            output = self.decode(encoder_output, target, encoder_padding_mask)
            output_indices = output.argmax(dim=-1)
            outputs[:, step] = output_indices[:, -1]
            
            if output_indices[0, -1].item() in self.stop_generation_tokens:
                break
            
        return outputs
    
    def beam(self, input: Tensor, beam_size: int = 5, max_len: int = 1000) -> tuple[Tensor, list]:
        x = input.unsqueeze(0)
        encoder_padding_mask, encoder_output = self.encode(x)
        
        encoder_padding_mask = encoder_padding_mask.repeat(beam_size, 1)
        encoder_output = encoder_output.repeat(beam_size, 1, 1)
        
        target = self.sos_token_idx * torch.ones(beam_size, 1).type_as(x).to(self.device)
        done = self.pad_token_idx * torch.ones(beam_size, max_len).type_as(x).to(self.device)
        done_probs = []
         
        beams_prob = [1] * beam_size
        
        for step in range(1, max_len):
            left = len(target)
            decoder_output = self.decode(encoder_output[:left], target, encoder_padding_mask[:left])
            decoder_output_probs = decoder_output.softmax(dim=-1)
            
            indices = decoder_output_probs.argsort(dim=-1, descending=True)[:, -1, :left].tolist()
            
            # list of tuples (pos, arg_idx, prob)
            candidates = [(pos, arg_idx, prob * decoder_output_probs[pos, -1, arg_idx]) for pos, (arg_ids, prob) in
                          enumerate(zip(indices, beams_prob)) for arg_idx in arg_ids]
            
            if step == 1:
                candidates = candidates[:beam_size]
            else:
                candidates = sorted(candidates, key=lambda candidate: candidate[2], reverse=True)[:left]
                
            target = torch.stack(
                [torch.cat([target[candidate[0]], torch.tensor([candidate[1]]).to(self.device)])
                 for candidate in candidates])
            
            is_completed = [candidate[1] in self.stop_generation_tokens for candidate in candidates]
            completed = target[is_completed]
            
            done[beam_size - left: beam_size - left + len(completed), :step + 1] = completed
            done_probs.extend([candidate[2] for value, candidate in zip(is_completed, candidates) if value])            
            
            target = target[~torch.tensor(is_completed)]
            beams_prob = [candidate[2] for value, candidate in zip(is_completed, candidates) if not value]
            
            if len(done_probs) == beam_size:
                break
            if step == max_len - 1:
                done[beam_size - left + len(completed):] = target
                done_probs.extend(beams_prob)
            
        return done, done_probs
        
    def init_weights(self):
        for parameter in self.parameters():
            if parameter.dim() > 1:
                torch.nn.init.xavier_uniform_(parameter)
    
    def get_state_dict(self) -> dict:
        state_dict = {
            'model': self.state_dict(),
            'config': {
                'dim': self.dim,
                'vocab_size': self.vocab_size,
                'num_heads': self.num_heads,
                'num_encoder_blocks': self.num_encoder_blocks,
                'num_decoder_blocks': self.num_decoder_blocks,
                'feed_forward_hidden_dim': self.feed_forward_hidden_dim,
                'sos_token_idx': self.sos_token_idx,
                'eos_token_idx': self.eos_token_idx,
                'pad_token_idx': self.pad_token_idx,
                'max_len': self.max_len,
                'dropout': self.dropout
            }
        }
        
        return state_dict
