import torch

from vision_transformer import VisionTransformer, ViTConfig


def run():
    config = ViTConfig(patch_size=16)
    model = VisionTransformer(config).cuda()
    model.flash_attention = True
    print(f'ViT | number of parameters: {model.num_params:,} | Number of positional embeddings parameters: {model.positional_embeddings.positional_embeddings.numel():,}')
    
    # forward pass
    images = torch.ones(4, 3, 224, 224).cuda()
    output = model(images)


if __name__ == '__main__':
    run()
