import torch
import torch.nn as nn
import torchvision.models as models

class ImageEncoder(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        base = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(base.children())[:-1])
        self.fc = nn.Linear(base.fc.in_features, out_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class SensorEncoder(nn.Module):
    def __init__(self, in_dim, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )
    def forward(self, x):
        return self.net(x)

class FusionModel(nn.Module):
    def __init__(self, sensor_dim, num_classes):
        super().__init__()
        self.img_enc = ImageEncoder(out_dim=256)
        self.sensor_enc = SensorEncoder(in_dim=sensor_dim, out_dim=64)
        self.fc = nn.Sequential(
            nn.Linear(256 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, img, sensors):
        fi = self.img_enc(img)
        fs = self.sensor_enc(sensors)
        f = torch.cat([fi, fs], dim=1)
        return self.fc(f)
