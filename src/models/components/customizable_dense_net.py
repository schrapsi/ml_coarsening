import torch
from torch import nn


class CustomizableDenseNet(nn.Module):

    def __init__(self, input_size=208, hidden_sizes=[512, 256], output_size=1, dropout=0.3, with_sigmoid=True, with_batch_norm=False):
        super().__init__()
        self.flatten = nn.Flatten()
        layers = []

        # First layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        if with_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        layers += [nn.ReLU(), nn.Dropout(p=dropout)]

        # Subsequent layers
        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            if with_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_sizes[i]))
            layers += [nn.ReLU(), nn.Dropout(p=dropout)]

        # Output layer
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
