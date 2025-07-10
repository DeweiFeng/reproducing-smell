# models/dataset.py
import os  # Path operations
import glob  # File pattern matching
import json  # JSON file parsing
import torch  # PyTorch tensors
import pandas as pd  # DataFrame for CSV reading
from torch.utils.data import Dataset  # Base class for datasets
from PIL import Image  # Image loading
import random  # Random choice for images

class SmellReproductionDataset(Dataset):
    def __init__(
        self,
        sensor_root: str,
        image_root: str,
        desc_path: str,
        transform=None,
        sensor_seq_len: int = 200
    ):
        self.sensor_root = sensor_root  # Directory containing sensor CSVs
        self.image_root = image_root  # Directory containing images
        self.transform = transform  # Optional image transforms
        self.sensor_seq_len = sensor_seq_len  # Fixed length for sensor

        # Load odor metadata from JSON
        with open(desc_path, 'r') as f:
            self.odor_info = json.load(f)['odors']

        # Build a list of all samples
        self.samples = []
        for odor in self.odor_info:
            name = odor['name']  # Odor name matches folder
            sensor_dir = os.path.join(sensor_root, name)  # Sensor folder
            image_dir = os.path.join(image_root, name)    # Image folder

            # Get all CSV files and JPG images
            sensor_paths = sorted(glob.glob(os.path.join(sensor_dir, '*.csv')))
            image_paths = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))

            for sensor_path in sensor_paths:
                self.samples.append({
                    'label': odor['id'] - 1,  # Zero-based label
                    'description': odor['long_description'],  # Text description
                    'sensor_path': sensor_path,  # CSV file path
                    'image_paths': image_paths  # All images for that odor
                })

    def __len__(self):
        return len(self.samples)  # Total number of sensor files

    def __getitem__(self, idx):
        item = self.samples[idx]  # Get sample info

        # Load sensor CSV into tensor
        df = pd.read_csv(item['sensor_path'])  # Read CSV
        sensor = torch.tensor(df.values, dtype=torch.float32)  # Convert to tensor
        # Pad or trim to fixed length
        if sensor.size(0) < self.sensor_seq_len:
            pad = torch.zeros(self.sensor_seq_len - sensor.size(0), sensor.size(1))
            sensor = torch.cat([sensor, pad], dim=0)
        else:
            sensor = sensor[:self.sensor_seq_len]

        # Load a random image for this odor
        img_path = random.choice(item['image_paths'])
        image = Image.open(img_path).convert('RGB')  # Read and convert to RGB
        if self.transform:
            image = self.transform(image)  # Apply transforms if provided

        return {
            'sensor': sensor,  # Tensor [seq_len, features]
            'image': image,    # PIL.Image
            'text': item['description'],  # Raw description string
            'label': item['label']  # Integer label
        }
