import torch
import torch.nn as nn
import torchvision.models.segmentation as models

class FCN(nn.Module):
    def __init__(self, num_classes):
        super(FCN, self).__init__()
        self.model = models.fcn_resnet50(pretrained=True)
        self.model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)

    def forward(self, x):
        return self.model(x)["out"]
