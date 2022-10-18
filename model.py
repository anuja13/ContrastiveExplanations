import torch.nn.functional as F
import torch.nn as nn
from torchvision.models.resnet import resnet50

class Resnet50(nn.Module):
    def __init__(self, num_classes=2):  # 1 for BCEwithLogits, 2 for CE and APL
        super(Resnet50, self).__init__()
        self.model = resnet50(pretrained=True)
        self.model.fc = nn.Sequential(
                        nn.Linear(2048, 128),
                        nn.ReLU(inplace=True),
                        nn.Linear(128, num_classes)
                        )
    

    def forward(self, x):
        x = self.model(x)
        return x
    
