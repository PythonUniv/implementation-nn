import pickle
import torch
from sentence_splitter import SentenceSplitter

from .transformer import Transformer
from .tokenizer import Tokenizer


class SplitText:
    def __init__(self, max_sentence_length: int = 160):
        self.sentence_splitter = SentenceSplitter(language='en')
        self.max_sentence_length = max_sentence_length
        
    def split(self, text: str) -> list[str]:
        sentences = self.sentence_splitter.split(text)
        inputs = []
        for sentence in sentences:
            if self.max_sentence_length < len(sentence):
                split_sentences = self.split_sentence(sentence)
                inputs.extend(split_sentences)
            else:
                inputs.append(sentence)
        return inputs
            
    def split_sentence(self, input: str) -> list[str]:
        sentences = []
        sentence = ''
        words = input.split()
        
        for word in words:
            if self.max_sentence_length < len(sentence) + len(word):
                if self.max_sentence_length < len(word):
                    if sentence:
                        sentences.append(sentence)
                        sentence = ''
                    sentences.extend([word[idx: idx + self.max_sentence_length] for idx in
                                      range(0, len(word), self.max_sentence_length)])
                else:
                    sentences.append(sentence)
                    sentence = word
            else:
                sentence += ' ' + word if sentence else word
        if sentence:
            sentences.append(sentence)
            
        return sentences
            

class TransformerInference:
    def __init__(
        self,
        input_tokenizer_path: str,
        output_tokenizer_path: str,
        device: str,
        model_path: str | None = None,
        model_state_dict_path: str | None = None,
        max_text_sentence_len: int = 160, 
    ):
        self.device = device
        
        if model_path:
            self.transformer = torch.load(model_path).to(device)
        
        if model_state_dict_path:
            state_dict = torch.load(model_state_dict_path)
            self.transformer = Transformer(**state_dict['config'])
            self.transformer.load_state_dict(state_dict['model'])
            self.transformer = self.transformer.to(device)
            
        self.transformer.eval()
        
        with open(input_tokenizer_path, 'rb') as input_tokenizer_file, \
        open(output_tokenizer_path, 'rb') as output_tokenizer_file:
            self.input_tokenizer: Tokenizer = pickle.load(input_tokenizer_file)
            self.output_tokenizer: Tokenizer = pickle.load(output_tokenizer_file)
            
        self.split_text = SplitText(max_text_sentence_len)
    
    def generate(self, input: str, max_len: int = 1000) -> str:
        input_tokenized = self.input_tokenizer.tokenize(input).to(self.device)
        output = self.transformer.generate(input_tokenized, max_len=max_len)[0]
        return self.output_tokenizer.tokens_to_str(output)
    
    def generate_many(self, input: str, max_len: int = 1000) -> str:
        sentences = [sentence for sentence in self.split_text.split(input) if sentence]
        generated = ' '.join(
            [self.generate(sentence, max_len) for sentence in sentences])
        return generated
    
    def beam(self, input, beam_size: int = 5, max_len: int = 1000) -> list[dict]:
        input_tokenized = self.input_tokenizer.tokenize(input).to(self.device)
        with torch.no_grad():
            outputs, beams = self.transformer.beam(input_tokenized, beam_size=beam_size, max_len=max_len)
        predicted = [
            {'text': self.output_tokenizer.tokens_to_str(output),
             'probability': probability.item()} for output, probability in zip(outputs, beams)
        ] 
        return predicted
