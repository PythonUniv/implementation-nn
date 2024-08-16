import argparse

from segformer import SegFormer, SegFormerConfig
from train import train


if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('-batch_size', type=int)
    argument_parser.add_argument('-epochs', type=int)
    argument_parser.add_argument('-lr', type=float)
    argument_parser.add_argument('-dataset_dir')
    argument_parser.add_argument('-image_size', default=224, type=int)
    argument_parser.add_argument('-device', default='cuda')
    argument_parser.add_argument('-save_dir', default='checkpoints')
    argument_parser.add_argument('-neptune_api_key', default=None)
    argument_parser.add_argument('-project', default=None)
    
    arguments = argument_parser.parse_args()
    
    config = SegFormerConfig(num_classes=24)
    segformer = SegFormer(config)
    
    if arguments.neptune_api_key is not None:
        import neptune
        neptune_run = neptune.init_run(project=arguments.project, api_token=arguments.neptune_api_key)
    else:
        neptune_run = None
    
    print(arguments)
    
    train(
        segformer, arguments.dataset_dir, arguments.batch_size, arguments.image_size,
        arguments.lr, arguments.epochs, arguments.device, arguments.save_dir, neptune_run=neptune_run)
