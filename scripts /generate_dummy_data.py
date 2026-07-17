import os
import numpy as np
from PIL import Image, ImageDraw
import pandas as pd

# Paths
data_dir = "data"
img_dir = os.path.join(data_dir, "images")
sensor_dir = os.path.join(data_dir, "sensors")
csv_path = os.path.join(data_dir, "labels.csv")

# Make sure folders exist
os.makedirs(img_dir, exist_ok=True)
os.makedirs(sensor_dir, exist_ok=True)

# Number of samples
n_samples = 5
sensor_dim = 10  # must match SENSOR_DIM in your model

rows = []

for i in range(1, n_samples + 1):
    img_name = f"sample{i}.jpg"
    sensor_name = f"sample{i}.npy"

    # --- Create a dummy image ---
    img = Image.new("RGB", (224, 224), color=(i*40 % 255, i*80 % 255, i*120 % 255))
    draw = ImageDraw.Draw(img)
    draw.text((50, 100), f"Face {i}", fill=(255, 255, 255))
    img.save(os.path.join(img_dir, img_name))

    # --- Create dummy sensor data ---
    sensor_vector = np.random.rand(sensor_dim).astype(np.float32)
    np.save(os.path.join(sensor_dir, sensor_name), sensor_vector)

    # --- Add entry to CSV ---
    label = (i - 1) % 3  # cycle through 0,1,2
    rows.append([img_name, sensor_name, label])

# Save labels.csv
df = pd.DataFrame(rows, columns=["image", "sensors", "label"])
df.to_csv(csv_path, index=False)

print("✅ Dummy dataset created!")
print(f"- {n_samples} images saved in {img_dir}")
print(f"- {n_samples} sensor files saved in {sensor_dir}")
print(f"- labels.csv created at {csv_path}")
