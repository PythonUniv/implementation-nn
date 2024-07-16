from resnet import Resnet
from train import train_resnet


def train():
    resnet = Resnet(num_classes=100)
    
    batch_size = 8
    epochs = 1
    train_resnet(resnet, batch_size, epochs)
    
    
if __name__ == '__main__':
    train()
