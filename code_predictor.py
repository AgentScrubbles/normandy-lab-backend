import torch
import torch.nn as nn
import torchvision.models as models

class CodePredictor(nn.Module):
    def __init__(self, num_classes_per_digit):
        super().__init__()
        self.backbone = models.mobilenet_v3_small(weights=None)
        in_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Identity()

        self.heads = nn.ModuleList([
            nn.Linear(in_features, num_classes)
            for num_classes in num_classes_per_digit
        ])

    def forward(self, x):
        features = self.backbone(x)
        outputs = [head(features) for head in self.heads]
        return outputs
