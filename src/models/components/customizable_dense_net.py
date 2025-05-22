import torch
from torch import nn


class CustomizableDenseNet(nn.Module):

    def __init__(self, input_size=208, hidden_sizes=[512, 256], output_size=1, dropout_rate=0.3, with_sigmoid=True):
        super().__init__()
        self.flatten = nn.Flatten()
        layers = [nn.Linear(input_size, hidden_sizes[0]), nn.ReLU(), nn.Dropout(p=dropout_rate)]

        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))

        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        if with_sigmoid:
            layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #x = self.flatten(x) old
        x = x.view(x.size(0), -1)  # Flatten the input tensor new
        x = self.model(x)
        return x


if __name__ == "__main__":
    _ = CustomizableDenseNet()
