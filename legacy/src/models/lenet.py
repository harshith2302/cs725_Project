import torch
import torch.nn as nn
import torch.nn.functional as F

# designing the neural net as prescribed in the paper.
class LeNet5(nn.Module):
    """
    LeNet-5 architecture as described in Lecun et al., 1998.
    """
    def __init__(self, activation='tanh'):
        super(LeNet5, self).__init__()
        self.activation_type = activation
        
        # C1: 6 filters, 5x5 kernel
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        # S2: Subsampling (AvgPool) 2x2
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        # C3: 16 filters, 5x5 kernel
        self.conv3 = nn.Conv2d(6, 16, kernel_size=5)
        # S4: Subsampling (AvgPool) 2x2
        self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
        # C5: 120 filters, 5x5 kernel
        self.conv5 = nn.Conv2d(16, 120, kernel_size=5)
        # F6: Fully connected 84 units
        self.fc6 = nn.Linear(120, 84)
        # Output: 10 units
        self.fc7 = nn.Linear(84, 10)

    def forward(self, x):
        if self.activation_type == 'tanh':
            act = torch.tanh
        else:
            act = F.relu

        # C1 -> Activation -> S2
        x = act(self.conv1(x))
        x = self.pool2(x)
        # C3 -> Activation -> S4
        x = act(self.conv3(x))
        x = self.pool4(x)
        # C5 -> Activation
        x = act(self.conv5(x))
        # Flatten
        x = torch.flatten(x, 1)
        # F6 -> Activation
        x = act(self.fc6(x))
        # Output
        x = self.fc7(x)
        return x
