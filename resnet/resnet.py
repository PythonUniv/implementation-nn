import torch
import torch.nn as nn


class Bottleneck(nn.Module):
    def __init__(self, in_channels: int, intermediate_channels: int, expansion: int, stride: int):
        super().__init__()
        
        self.in_channels = in_channels
        self.intermediate_channels = intermediate_channels
        self.expansion = expansion
        self.stride = stride
        
        self.output_channels = expansion * intermediate_channels
        
        if self.in_channels == self.output_channels:
            self.same_feature_map = True
        else:
            self.same_feature_map = False
            self.projection_layer = nn.Sequential(
                nn.Conv2d(
                    self.in_channels, self.output_channels, kernel_size=1,
                    stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(self.output_channels)
            )
        
        self.relu = nn.ReLU()
        self.conv_1 = nn.Conv2d(self.in_channels, self.intermediate_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.batch_norm_1 = nn.BatchNorm2d(self.intermediate_channels)
        self.conv_2 = nn.Conv2d(
            self.intermediate_channels, self.intermediate_channels, kernel_size=3,
            stride=stride, padding=1, bias=False)
        self.batch_norm_2 = nn.BatchNorm2d(self.intermediate_channels)
        self.conv_3 = nn.Conv2d(
            self.intermediate_channels, self.output_channels, kernel_size=1,
            stride=1, padding=0, bias=False
        )
        self.batch_norm_3 = nn.BatchNorm2d(self.output_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Args:
                x (torch.Tensor): tensor of shape (B, in_channels, H, W)
            Returns:
                torch.Tensor of shape (B, intermediate_channels * expansion, H, W)
        """
        
        x_conv = self.relu(self.batch_norm_1(self.conv_1(x)))
        x_conv = self.relu(self.batch_norm_2(self.conv_2(x_conv)))
        x_conv = self.batch_norm_3(self.conv_3(x_conv))
        if self.same_feature_map:
            x_conv = x_conv + x
        else:
            x_conv = x_conv + self.projection_layer(x)
        return self.relu(x_conv)


class ResnetModule(nn.Module):
    def __init__(self, in_channels: int, intermediate_channels: int, expansion: int, stride: int, num_layers: int):
        super().__init__()
        
        self.in_channels = in_channels
        self.intermediate_channels = intermediate_channels
        self.expansion = expansion
        self.stride = stride
        
        self.output_channels = expansion * intermediate_channels
        
        self.module = nn.Sequential(
            Bottleneck(in_channels, intermediate_channels, expansion, stride=stride),
            *[Bottleneck(self.output_channels, intermediate_channels, expansion, stride=1) for idx in range(num_layers - 1)]
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)


class Resnet(nn.Module):
    def __init__(self, in_channels: int = 3, modules_config: dict | None = None, num_classes: int = 1000):
        super().__init__()
        
        # modules_config with keys 'expansion', 'intermediate_channels', 'repetitions' 
        if modules_config is None:
            modules_config = self.resnet152_config()
        
        self.in_channels = in_channels
        self.conv_1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm_1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.resnet_modules = nn.ModuleList()
        self.resnet_modules.append(
            ResnetModule(
                64, modules_config['intermediate_channels'][0], expansion=modules_config['expansion'],
                stride=1, num_layers=modules_config['repetitions'][0])
        )
        for idx in range(1, 4):
            in_c = self.resnet_modules[idx - 1].output_channels
            self.resnet_modules.append(
                ResnetModule(
                    in_c, modules_config['intermediate_channels'][idx],
                    expansion=modules_config['expansion'], stride=2, num_layers=modules_config['repetitions'][idx])
            )
        
        c_out = self.resnet_modules[-1].output_channels
                
        self.average_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_1 = nn.Linear(c_out, num_classes)

    @staticmethod
    def resnet152_config() -> dict:
        return {
            'expansion': 4,
            'intermediate_channels': [64, 128, 256, 512],
            'repetitions': [3, 8, 36, 3]
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.batch_norm_1(self.conv_1(x)))
        x = self.max_pool(x)
        for module in self.resnet_modules:
            x = module(x)
        x = self.average_pool(x)
        return self.fc_1(torch.flatten(x, start_dim=1))
