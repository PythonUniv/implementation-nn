from model import DiffusionModel, DenoisingDiffusionConfig
from train import train, TrainConfig


def run_train():
    train_config = TrainConfig(epochs=3, batch_size=128, lr=1e-4, dataset_root=r'C:\Users\Ноутбук\Desktop\enviroment\diffusion_model\dataset', auto_cast=False)
    config = DenoisingDiffusionConfig()
    model = DiffusionModel(config)
    
    train(model, train_config)


if __name__ == '__main__':
    run_train()
