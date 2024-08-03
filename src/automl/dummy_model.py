from torch import nn
import math
from torchvision import models, transforms
from torchvision.models import vgg16


class DummyNN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        num_neurons = 128
        layers = [nn.Flatten(),
                  nn.Linear(input_size, num_neurons),
                  nn.ReLU(),
                  nn.Linear(num_neurons, output_size)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class DummyCNN(nn.Module):

    def __init__(self,
                 input_channels: int,
                 hidden_channels: int,
                 output_channels: int,
                 image_width: int) -> None:
        super(DummyCNN, self).__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        feature_size = self.calculateFeatureSize(
            hidden_channels, image_width, 2, 2)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_size, output_channels)
        )

    def calculateFeatureSize(self, hidden_channels, width, pool_size1, pool_size2):
        size = width
        size = math.floor((size - (pool_size1 - 1) - 1) / pool_size1 + 1)
        size = math.floor((size - (pool_size2 - 1) - 1) / pool_size2 + 1)
        return size * hidden_channels * size

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x


class VGG16(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, mean: float, std: float):
        super().__init__()
        self.normalize = transforms.Normalize(mean=mean, std=std)
        self.resize = transforms.Resize((224, 224))
        ## buraya generic value'lar yollanacak, automl'de config'ten çektiğim data trasnformerları, buraya yollanacak
        self.model = vgg16()
        self.model.features[0] = nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=(3, 3),
                                           stride=(1, 1), padding=(1, 1))
        self.model.classifier[6] = nn.Linear(in_features=4096, out_features=output_channels, bias=True)

    def forward(self, x): ## validationa normalize ve resize veya augmentasyonlar eklenmememeli, if else koyılmalı alta
        return self.model(self.normalize(self.resize(x)))
