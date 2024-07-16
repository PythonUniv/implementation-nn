from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard.writer import SummaryWriter
from bert.tokenizer import tokenizer
from tokenizers import Tokenizer

from bert.bert import Bert
from bert.dataset_db import BertDatasetDatabase
from bert.dataset import make_dataset, BertDataset
from bert.pretraining import BertPretrain
from bert.tokenizer import tokenizer
from bert.train import BertTrainer


tokenizer: Tokenizer

    
if __name__ == '__main__':
    create_dataset = False
    train = True
    
    database_path = r'C:\Users\Ноутбук\Desktop\enviroment\bert\dataset\database\database.db'
    raw_files_dir = r'C:\Users\Ноутбук\Desktop\enviroment\bert\dataset\raw_files'
    save_dir = r'C:\Users\Ноутбук\Desktop\enviroment\bert\save'
    log_dir = r'C:\Users\Ноутбук\Desktop\enviroment\bert\logs'
    
    device = 'cuda'
    
    database = BertDatasetDatabase(database_path)
    database.connect()
    
    if create_dataset:
        database.init_table()
        make_dataset(database_path, raw_files_dir)
        
    if not train:
        exit(0)
    
    epochs = 3
    batch_size = 16
    train_part = 0.99
    lr = 1e-4
    
    dataset = BertDataset(bert_database=database)
    train_dataset, valid_dataset = random_split(dataset, lengths=[train_part, 1 - train_part])
    train_data_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    valid_data_loader = DataLoader(valid_dataset, batch_size, shuffle=True)
    
    bert_config = {
        'dim': 1024,
        'hidden_dim': 2048,
        'num_heads': 8,
        'num_blocks': 6,
        'vocab_size': tokenizer.get_vocab_size(),
        'max_len': 10000,
        'device': device
    }
    
    bert_pretrain_config = {
        'ml_hidden': 1024,
        'ns_hidden': 1024,
        'device': device,
    }   
    
    bert = Bert(**bert_config)
    bert_pretrain = BertPretrain(bert, **bert_pretrain_config)
    
    summary_writer = SummaryWriter(log_dir)
    bert_trainer = BertTrainer(
        bert_pretrain, tokenizer, device, [0, 100, 101, 102, 103], summary_writer=summary_writer)
    bert_trainer.train(train_data_loader, valid_data_loader, epochs, save_dir, lr)
