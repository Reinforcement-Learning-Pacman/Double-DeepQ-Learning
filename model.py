import numpy as np
import torch
import torch.nn as nn


class DQNModel(nn.Module):
    def __init__(self, input_shape: tuple, n_actions: int) -> None:
        super(DQNModel, self).__init__()
        # cnn extract features from images of the game
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # fully connected
        self.fc = nn.Sequential(
            nn.Linear(self._get_conv_output(input_shape), 512), # input: feature vector from CNN
            nn.ReLU(),
            nn.Linear(512, n_actions) 
        )

    def _get_conv_output(self, shape: tuple) -> int:
        batch = torch.zeros(1, *shape) 
        conv_out = self.conv(batch)
        return int(np.prod(conv_out.size()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float() / 255.0
        conv_out = self.conv(x)
        conv_out = conv_out.view(x.size()[0], -1) # flatten tensor multi to 1 dimension
        return self.fc(conv_out)
