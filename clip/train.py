from clip import DistilBertEncoder, VisionTransformerEncoder, CLIP, train
from dataset import get_data_loaders


if __name__ == '__main__':
    distil_bert_encoder = DistilBertEncoder()
    vision_transformer_encoder = VisionTransformerEncoder()
    
    clip = CLIP(
        distil_bert_encoder, vision_transformer_encoder,
        text_encoder_dim=768, vision_encoder_dim=768, proj_dim=256)
    
    train_loader, val_loader = get_data_loaders(batch_size=128, num_workers=4)
    train(clip, train_loader, val_loader, epochs=1)
