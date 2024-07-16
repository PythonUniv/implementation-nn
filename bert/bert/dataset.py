import os
from pathlib import Path
from itertools import pairwise
import random
from typing import Generator, AnyStr
from sentence_splitter import SentenceSplitter
from torch.utils.data import Dataset
from tqdm import tqdm

from .dataset_db import BertDatasetDatabase


def load_chunks(path: AnyStr, chunk_size: int = 1000000) -> Generator[str, None, None]:
    with open(path, 'rb') as file:
        while chunk := file.read(chunk_size):
            yield chunk.decode(errors='ignore')
            

def make_dataset(database_path, read_dir: str, min_sentence_len: int = 10, max_sentence_len: int = 140):
    database = BertDatasetDatabase(database_path)
    database.connect()
    database.init_table()
    
    read_dir_path = Path(read_dir)
    
    read_paths = list(read_dir_path.glob('*.txt'))
    random.shuffle(read_paths)
    
    sentence_splitter = SentenceSplitter('en')
    
    for read_path in tqdm(read_paths):
        chunks = load_chunks(read_path, chunk_size=int(1e7))
        for chunk in chunks:
            sentences = sentence_splitter.split(chunk)
            sentence_pairs = [(sentence_1, sentence_2) for sentence_1, sentence_2 in  pairwise(sentences)
                              if min_sentence_len <= len(sentence_1) <= max_sentence_len and min_sentence_len <= len(sentence_2) <= max_sentence_len]
            database.add_sentence_pairs(sentence_pairs)
    database.close()
    
    
class BertDataset(Dataset):
    def __init__(self, bert_database: BertDatasetDatabase):
        self.bert_database = bert_database
        
    def __len__(self) -> int:
        return len(self.bert_database)
    
    def __getitem__(self, idx: int) -> tuple[str, str]:
        return self.bert_database.get(idx)
    
    
if __name__ == '__main__':
    database_dir = r'C:\Users\Ноутбук\Desktop\enviroment\bert\dataset\database.db'
    read_dir = r'C:\Users\Ноутбук\Desktop\enviroment\bert\dataset\files'
    make_dataset(database_dir, read_dir)
