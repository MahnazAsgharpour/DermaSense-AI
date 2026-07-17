import torchvision.transforms as T
import torch

# Image transforms for training and validation
def get_image_transforms(train=True):
    if train:
        return T.Compose([
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(),     # small augmentation
            T.RandomRotation(10),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
        ])
    else:
        return T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
        ])

# Sensor transforms (normalize scalar inputs)
class SensorTransform:
    """
    Normalize sensor vectors (e.g., hydration, elasticity, pH).
    You can set mean and std values based on your dataset.
    """
    def __init__(self, mean=None, std=None):
        # Example: if unknown, assume mean=0, std=1 (no normalization)
        self.mean = torch.tensor(mean) if mean is not None else None
        self.std = torch.tensor(std) if std is not None else None

    def __call__(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / self.std
        return x
