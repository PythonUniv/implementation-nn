import os
import multiprocessing as mp
import numpy as np
from typing import Literal
from dotenv import load_dotenv
from datasets import load_dataset, IterableDataset
import tiktoken
from tqdm import tqdm


load_dotenv()


tokenizer_name = 'gpt2'
print(f'Using tokenizer: {tokenizer_name}')
tokenizer = tiktoken.get_encoding(tokenizer_name)


def tokenize(document: dict) -> np.ndarray:
    tokens = [tokenizer.eot_token]
    tokens.extend(tokenizer.encode_ordinary(document['text']))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all(), 'Text vocabulary too large for np.uint16.'
    assert (tokens_np < 2 ** 16).all(), 'Text vocabulary too large for np.uint16.'
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16


def create_dataset_shards(
    dataset: IterableDataset, directory: str, split_name: Literal['train', 'val'] = 'train',
    shard_size: int = 2 ** 25, num_processes: int | None = None
):
    os.makedirs(directory, exist_ok=True)
    num_processes = max(1, os.cpu_count() // 2) if num_processes is None else num_processes
    print(f'Tokenizing using {num_processes} processes.')
    
    with mp.Pool(num_processes) as pool:
        shard_idx = 0
        num_pushed = 0
        buffer = np.empty((shard_size,), dtype=np.uint16)
        tqdm_bar = tqdm(total=shard_size, desc=f'Shard: {shard_idx}', unit='tokens')
        
        for tokens in pool.imap(tokenize, dataset, chunksize=16):
            if num_pushed + len(tokens) <= shard_size:
                buffer[num_pushed: num_pushed + len(tokens)] = tokens
                num_pushed += len(tokens)
                tqdm_bar.update(len(tokens))
                
            else:
                remainder = shard_size - num_pushed
                buffer[num_pushed: num_pushed + remainder] = tokens[:remainder]
                tqdm_bar.update(remainder)
                shard_file_name = os.path.join(directory, f'shard_{split_name}_{shard_idx}.npy')
                np.save(shard_file_name, buffer)
                tqdm_bar.set_description_str(f'Shard {shard_idx} is saved.')
                
                shard_idx += 1

                num_pushed = len(tokens) - remainder
                buffer[:num_pushed] = tokens[remainder: remainder + num_pushed]
                tqdm_bar = tqdm(total=shard_size, desc=f'Shard: {shard_idx}', unit='tokens')
                tqdm_bar.update(num_pushed)
                
        if num_pushed != 0:
            completeness = 'complete' if num_pushed == shard_size else 'incomplete'
            tqdm_bar.set_description(f'The last {completeness} shard {shard_idx} is saved.')
            shard_file_name = os.path.join(directory, f'shard_{split_name}_{shard_idx}.npy')
            np.save(shard_file_name, buffer[:num_pushed])


if __name__ == '__main__':
    path = 'nampdn-ai/tiny-textbooks'
    dataset = load_dataset(path, split='train', token=os.environ['HF_ACCESS_TOKEN'])
    directory = os.path.join(os.path.dirname(__file__), 'shards')
    
    create_dataset_shards(dataset, directory, shard_size=int(5e7))
