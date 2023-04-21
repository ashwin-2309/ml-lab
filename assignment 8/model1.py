# 2020BCS-017 Ashwin Kumar Singh

import torch
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        # Convolutional layer 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5)

        # Maxpooling layer 1
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=1)

        # Convolutional layer 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)

        # Maxpooling layer 2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional layer 3
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1)

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=64*12*12, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # Convolutional layer 1
        x = self.conv1(x)
        x = nn.ReLU()(x)

        # Maxpooling layer 1
        x = self.pool1(x)

        # Convolutional layer 2
        x = self.conv2(x)
        x = nn.ReLU()(x)

        # Maxpooling layer 2
        x = self.pool2(x)

        # Convolutional layer 3
        x = self.conv3(x)
        x = nn.ReLU()(x)

        # Flatten the output
        x = torch.flatten(x, start_dim=1)

        # Fully connected layers
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)
        x = self.fc3(x)

        return x


model = MyModel()
print(model)

# This model has a total of 308,394 learnable parameters, including biases.

# The number of parameters at each layer is as follows:

#     CNN-layer-1: 416 parameters (16 * (5*5) + 16)
#     CNN-layer-2: 4,640 parameters (32 * (3*3) * 16 + 32)
#     CNN-layer-3: 2,080 parameters (64 * 1 * 32 + 64)
#     Fully connected layer: 2,097,290 parameters ((641212) * 256 + 256 + 256 * 128 + 128 + 128 * 10 + 10)

# The number of features at each layer is as follows:

#     CNN-layer-1: 16 feature maps
#     Sub-sampling layer-1: 16 feature maps (same as CNN-layer-1)
#     CNN-layer-2: 32 feature maps
#     Sub-sampling layer-2: 32 feature maps (same as CNN-layer-2)
#     CNN-layer-3: 64 feature maps
#     Fully-connected layer: 10 output features (for the 10 classes in the dataset)
