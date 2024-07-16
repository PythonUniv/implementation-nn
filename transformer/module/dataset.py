from torch import Tensor
from torch.utils.data import Dataset

from .tokenizer import Tokenizer


class TranslationDataset(Dataset):
    def __init__(
        self,
        source_lang_sentences: list[str],
        target_lang_sentences: list[str],
        source_tokenizer: Tokenizer,
        target_tokenizer: Tokenizer
    ):
        super().__init__()
        self.source_lang_sentences = source_lang_sentences
        self.target_lang_sentences = target_lang_sentences
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        
    def __len__(self) -> int:
        return len(self.source_lang_sentences)
    
    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return (self.source_tokenizer.tokenize(self.source_lang_sentences[index]),
                self.target_tokenizer.tokenize(self.target_lang_sentences[index]))
    
    
def get_sentences(path_1: str, path_2: str, max_sentence_length: int = 160) -> tuple[list[str]]:
    lang_data_1, lang_data_2 = [], []
    with open(path_1, 'rb') as file_1, open(path_2, 'rb') as file_2:
        for line_1, line_2 in zip(file_1, file_2):
            if (
                1 <= len(sentence_1 := line_1.strip().decode()) <= max_sentence_length and
                1 <= len(sentence_2 := line_2.strip().decode()) <= max_sentence_length
            ):
                lang_data_1.append(sentence_1)
                lang_data_2.append(sentence_2)
    return lang_data_1, lang_data_2


def load_dataset(paths: list[tuple[str, str]], max_sentence_length: int = 160) -> tuple[list[str]]:
    source_sentences = []
    target_sentences = []
    for source_path, target_path in paths:
        sentences_1, sentences_2 = get_sentences(source_path, target_path, max_sentence_length)
        source_sentences.extend(sentences_1)
        target_sentences.extend(sentences_2)
    return source_sentences, target_sentences
