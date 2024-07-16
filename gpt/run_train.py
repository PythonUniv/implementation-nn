import os
from train import train


if __name__ == '__main__':
    data_root = os.path.join(os.path.dirname(__file__), 'shards')
    total_batch_size = 10 * 100 * 100
    
    train(data_root, batch_size=10, max_seq_len=100, total_batch_size=total_batch_size)
