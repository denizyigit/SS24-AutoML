from torch import nn
from torchvision.models import ResNet18_Weights
from torchvision import models


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


class dummyCNN(nn.Module):
    def __init__(self,
                 input_shape: int,
                 hidden_units: int,
                 output_shape: int) -> None:
        super().__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,  # or (3,3)
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      padding=1,
                      stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
            # tries to reduce the size of the image by half because of the kernel size is 2x2
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 7 * 7,
                      out_features=output_shape),
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classier(x)

        return x


class resNet18_Pretrained(nn.Module):
    def __init__(self, num_classes, number_of_channels):
        super().__init__()
        self.model = models.resnet18(weights=ResNet18_Weights)
        num_ftrs = self.model.fc.in_features
        if number_of_channels != 3:
            self.model.conv1 = nn.Conv2d(number_of_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                                         bias=False)
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)
