import os
import glob
import json
import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import random


class SmellReproductionDataset(Dataset):
    def __init__(self, sensor_root, image_root, desc_json, transform=None, sensor_seq_len=200):
        self.sensor_root = sensor_root
        self.image_root = image_root
        self.transform = transform
        self.sensor_seq_len = sensor_seq_len

        # Load descriptions
        with open(desc_json, 'r') as f:
            self.odor_info = json.load(f)['odors']

        # Build list of all samples
        self.samples = []
        for odor in self.odor_info:
            name = odor['name']
            sensor_dir = os.path.join(sensor_root, name)
            image_dir = os.path.join(image_root, name)

            print(sensor_dir)
            print(image_dir)

            sensor_paths = sorted(glob.glob(os.path.join(sensor_dir, '*.csv')))
            image_paths = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))

            for sensor_path in sensor_paths:
                self.samples.append({
                    'odor_id': odor['id'] - 1,  # zero-based label
                    'odor_name': name,
                    'description': odor['long_description'],
                    'sensor_path': sensor_path,
                    'image_paths': image_paths
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        # Load sensor data
        df = pd.read_csv(item['sensor_path'])
        sensor = torch.tensor(df.values, dtype=torch.float32)

        # Pad or trim to fixed sequence length
        if sensor.shape[0] < self.sensor_seq_len:
            pad = torch.zeros(self.sensor_seq_len - sensor.shape[0], sensor.shape[1])
            sensor = torch.cat([sensor, pad], dim=0)
        else:
            sensor = sensor[:self.sensor_seq_len]

        # Load a random image from image_paths
        img_path = random.choice(item['image_paths'])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return {
            'sensor': sensor,                             # [seq_len, num_features]
            'image': image,                               # Transformed image tensor
            'text': item['description'],                  # Smell description string
            'label': item['odor_id'],                     # Integer label (0–11)
            'odor_name': item['odor_name'],               # e.g., 'banana'
        }


if __name__ == "__main__":
    from torchvision import transforms
    from torch.utils.data import DataLoader

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = SmellReproductionDataset(
        sensor_root='../smell_data',
        image_root='../images',
        desc_json='../smell_descriptions.json',
        transform=transform
    )

    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    for batch in loader:
        print(batch['sensor'].shape)  # [B, 200, num_features]
        print(batch['image'].shape)   # [B, 3, 224, 224]
        print(batch['text'])          # List of strings
        break
