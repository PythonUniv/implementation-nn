import os
from dotenv import load_dotenv

from module.transformer import Transformer
from module.train import train
from module.dataset import load_dataset


load_dotenv()


os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


pad_token_idx = 0
sos_token_idx = 1
eos_token_idx = 2


device = 'cuda'


if __name__ == '__main__':
    save_folder = r'C:\Users\Ноутбук\Desktop\enviroment\transformer\model'
    vocab_size = 5000
    max_sentence_length = 80
    
    dataset_paths = [
        (r'C:\Users\Ноутбук\Desktop\enviroment\english_ukrainian_dataset\Tatoeba.en-uk.en', r'C:\Users\Ноутбук\Desktop\enviroment\english_ukrainian_dataset\Tatoeba.en-uk.uk')
    ]
    
    en_dataset, uk_dataset = load_dataset(dataset_paths, max_sentence_length=max_sentence_length)
    
    model_params = {
        'dim': 64,
        'vocab_size': vocab_size,
        'num_heads': 8,
        'num_encoder_blocks': 6,
        'num_decoder_blocks': 6,
        'feed_forward_hidden_dim': 64,
        'sos_token_idx': sos_token_idx,
        'eos_token_idx': eos_token_idx,
        'pad_token_idx': pad_token_idx,
        'device': device
    }
    
    train_params = {
        'source_sentences': en_dataset,
        'target_sentences': uk_dataset,
        'epochs': 3,
        'vocab_size': vocab_size,
        'batch_size': 256,
        'save_folder': save_folder,
        'max_lr': 2.8e-4,
        'log_dir': r'C:\Users\Ноутбук\Desktop\enviroment\transformer\logs',
        'wandb_logs_period': 100,
        'wandb_api_key': os.environ['WANDB_API_KEY']
    }
    
    model = Transformer(**model_params).to(device)
    train(model=model, **train_params)
