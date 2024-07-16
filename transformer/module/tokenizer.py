import pickle
from typing import Iterator
import torch
from tokenizers import Tokenizer as TokenizersTokenizer, AddedToken
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.processors import TemplateProcessing


pad_token = '[PAD]'
sos_token = '[SOS]'
eos_token = '[EOS]'
unk_token = '[UNK]'


class Tokenizer():
    def __init__(
        self,
        vocab_size: int,
        pad_token: str = pad_token,
        sos_token: str = sos_token,
        eos_token: str = eos_token,
        unk_token: str = unk_token
    ):
        self.tokenizer = TokenizersTokenizer(BPE(unk_token=unk_token))
        
        self.vocab_size = vocab_size
        self.pad_token = pad_token
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        
        self.tokenizer.pre_tokenizer = Whitespace()
        self.tokenizer.enable_padding(pad_id=0, pad_token=pad_token)
        
        self.tokenizer.post_processor = TemplateProcessing(
            single=f'{sos_token} $A {eos_token}',
            pair='$A $B',
            special_tokens=[(pad_token, 0), (sos_token, 1), (eos_token, 2), (unk_token, 3)]
        )
        
    def train(self, iterator: Iterator):
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=5,
            special_tokens=[token for token in (self.pad_token, self.sos_token, self.eos_token, self.unk_token)])
        self.tokenizer.train_from_iterator(iterator=iterator, trainer=trainer)
    
    def save(self, path: str):
        with open(path, 'wb+') as file:
            pickle.dump(self, file=file)
    
    @staticmethod
    def from_file(path: str) -> 'Tokenizer':
        with open(path, 'rb') as file:
            return pickle.load(file)
    
    def tokenize(self, text: str) -> torch.Tensor:
        return torch.tensor(self.tokenizer.encode(text).ids)
    
    def tokenize_batch(self, samples: list[str]) -> torch.Tensor:
        return torch.tensor(self.tokenizer.encode_batch(samples).ids)
    
    def tokens_to_str(self, ids: torch.Tensor) -> str:
        return self.tokenizer.decode(ids.tolist())
        
    def batch_tokens_to_str(self, ids: torch.Tensor) -> str:
        return self.tokenizer.decode_batch(ids.tolist())
    

if __name__ == '__main__':
    tokenizer = Tokenizer(vocab_size=10000)
