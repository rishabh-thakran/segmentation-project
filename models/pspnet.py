import torch
import torch.nn as nn

class PSPNet(nn.Module):
    def __init__(self, num_classes):
        super(PSPNet, self).__init__()
        self.encoder = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.ReLU())
        self.decoder = nn.Sequential(nn.Conv2d(64, num_classes, 3, padding=1))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
