import os
import glob
import json
import torch
import pandas as pd
from torch.utils.data import Dataset
from torch import Tensor


class SmellOnlyDataset(Dataset):
    """
    PyTorch Dataset that returns only the gas-nose sequence and its label.
    """
    def __init__(
        self,
        sensor_root: str,
        desc_path: str,
        sensor_seq_len: int = 200
    ):
        super().__init__()
        self.sensor_root = sensor_root
        self.sensor_seq_len = sensor_seq_len

        # Load odor metadata (assume UTF-8 encoding)
        with open(desc_path, 'r', encoding='utf-8') as f:
            odor_info = json.load(f)['odors']

        # Build list of (sensor_path, label)
        self.samples = []
        for odor in odor_info:
            name = odor['name']
            label = odor['id'] - 1  # zero-based labels
            sensor_dir = os.path.join(sensor_root, name)
            for path in sorted(glob.glob(os.path.join(sensor_dir, '*.csv'))):
                self.samples.append((path, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        path, label = self.samples[idx]

        # 1) Load CSV -> Tensor [N, F]
        df = pd.read_csv(path)
        sensor = torch.tensor(df.values, dtype=torch.float32)

        # 2) Pad or trim to fixed length [seq_len, F]
        seq_len, feat_dim = sensor.shape
        if seq_len < self.sensor_seq_len:
            padding = torch.zeros(self.sensor_seq_len - seq_len, feat_dim)
            sensor = torch.cat([sensor, padding], dim=0)
        else:
            sensor = sensor[:self.sensor_seq_len]

        return {
            'sensor': sensor,                     # Tensor [sensor_seq_len, F]
            'label': torch.tensor(label, dtype=torch.long)
        }