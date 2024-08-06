import argparse

from clip import DistilBertEncoder, VisionTransformerEncoder, CLIP, train
from dataset import get_data_loaders


if __name__ == '__main__':
    distil_bert_encoder = DistilBertEncoder()
    vision_transformer_encoder = VisionTransformerEncoder()
    
    clip = CLIP(
        distil_bert_encoder, vision_transformer_encoder,
        text_encoder_dim=768, vision_encoder_dim=768, proj_dim=256)
    
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('-batch_size', default=128)
    argument_parser.add_argument('-num_workers', default=0)
    argument_parser.add_argument('-epochs', default=1)
    arguments = argument_parser.parse_args()
    
    train_loader, val_loader = get_data_loaders(batch_size=arguments.batch_size, num_workers=arguments.num_workers)
    
    train(clip, train_loader, val_loader, epochs=arguments.epochs)
