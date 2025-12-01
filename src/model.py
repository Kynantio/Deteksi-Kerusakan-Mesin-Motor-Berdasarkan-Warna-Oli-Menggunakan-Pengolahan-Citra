import torch.nn as nn
import torchvision.models as models

class ResNet50Transfer(nn.Module):
    def __init__(self, num_classes=4):
        super(ResNet50Transfer, self).__init__()
        from torchvision.models import resnet50, ResNet50_Weights
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)