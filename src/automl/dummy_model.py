from torch import nn
import math


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
                 image_width:int) -> None:
        super(DummyCNN, self).__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        feature_size = self.calculateFeatureSize(hidden_channels,image_width, 2, 2)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_size, output_channels)
        )

    def calculateFeatureSize(self,hidden_channels,width, pool_size1,pool_size2):
        size = width
        size = math.floor((size - (pool_size1 - 1) - 1) / pool_size1 + 1)
        size = math.floor((size - (pool_size2 - 1) - 1) / pool_size2 + 1)
        return size*hidden_channels*size



    def forward (self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x

