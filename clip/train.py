import argparse

from clip import DistilBertEncoder, VisionTransformerEncoder, CLIP, train
from dataset import get_data_loaders


if __name__ == '__main__':
    distil_bert_encoder = DistilBertEncoder()
    vision_transformer_encoder = VisionTransformerEncoder()
    
    clip = CLIP(
        distil_bert_encoder, vision_transformer_encoder,
        text_encoder_dim=768, vision_encoder_dim=768, proj_dim=256)
    
    clip.compile()
    
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('-batch_size', default=128, type=int)
    argument_parser.add_argument('-num_workers', default=0, type=int)
    argument_parser.add_argument('-epochs', default=1, type=int)
    argument_parser.add_argument('-checkpoint_dir', default=None)
    argument_parser.add_argument('-neptune_api_key', default=None)
    argument_parser.add_argument('-project', default=None)
    
    arguments = argument_parser.parse_args()
    
    print(f'Training CLIP with summary number of parameters: {sum(parameter.numel() for parameter in clip.parameters()):,}')
    print(f'Number of trainable parameters: {sum(parameter.numel() for parameter in clip.parameters() if parameter.requires_grad):,}')
    
    if argument_parser.neptune_api_key is not None and argument_parser.project is not None:
        import neptune
        neptune_run = neptune.init_run(project=argument_parser.project, api_token=argument_parser.neptune_api_key)
    else:
        neptune_run = None
    
    train_loader, val_loader = get_data_loaders(batch_size=arguments.batch_size, num_workers=arguments.num_workers)
    
    train(clip, train_loader, val_loader, epochs=arguments.epochs, neptune_run=neptune_run, checkpoint_dir=argument_parser.checkpoint_dir)
