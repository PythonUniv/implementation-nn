import os
import argparse

from segformer import SegFormer, SegFormerConfig
from dataset import download
from train import train


if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('-batch_size', type=int)
    argument_parser.add_argument('-epochs', type=int)
    argument_parser.add_argument('-lr', type=float)
    argument_parser.add_argument('-dataset_dir')
    argument_parser.add_argument('-save_dir')
    argument_parser.add_argument('-image_size', default=224, type=int)
    argument_parser.add_argument('-device', default='cuda')
    argument_parser.add_argument('-save_dir', default='checkpoints')
    argument_parser.add_argument('-download', action='store_true')
    
    arguments = argument_parser.parse_args()
    
    config = SegFormerConfig()
    segformer = SegFormer(config)
    
    if arguments.download:
        download(arguments.dataset_dir)
        
    parquets = [os.path.join(arguments.dataset_dir, 'data', name) for name in os.listdir(arguments.dataset_dir)]
    num_train_parquets = int(0.8 * len(parquets))
    train_parquets = parquets[:num_train_parquets]
    val_parquets = parquets[num_train_parquets:]
    train(
        segformer, train_parquets, val_parquets, arguments.batch_size, arguments.image_size,
        arguments.lr, arguments.epochs, arguments.device, arguments.save_dir)
