from torch import nn


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