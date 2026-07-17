import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import pandas as pd

class MultimodalSkinDataset(Dataset):
    def __init__(self, csv_file, img_dir, sensor_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.sensor_dir = sensor_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Load image
        img_path = f"{self.img_dir}/{row['image']}"
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        # Load sensor vector
        sensor_path = f"{self.sensor_dir}/{row['sensors']}"
        sensors = np.load(sensor_path)
        sensors = torch.tensor(sensors, dtype=torch.float32)
        label = torch.tensor(row['label'], dtype=torch.long)
        return img, sensors, label
