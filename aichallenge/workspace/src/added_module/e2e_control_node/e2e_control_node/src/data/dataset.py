from pathlib import Path
import numpy as np
import h5py
import hdf5plugin

import torch
from torch.utils.data import Dataset

class DrivingDataset(Dataset):
    """
    複数のHDF5ファイルをまとめて扱い、辞書形式でデータを返すカスタムDatasetクラス
    """
    def __init__(self, data_dir: str, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform 
        self.h5_files = sorted(list(self.data_dir.glob('*.h5')))
        
        if not self.h5_files:
            raise FileNotFoundError(f"No .h5 files found in directory: {data_dir}")

        self.file_info = []
        self.total_samples = 0
        for h5_path in self.h5_files:
            with h5py.File(h5_path, 'r') as f:
                num_samples = len(f['images'])
                self.file_info.append({'path': h5_path, 'num_samples': num_samples, 'start_index': self.total_samples})
                self.total_samples += num_samples
        
        print(f"Loaded {len(self.h5_files)} files with a total of {self.total_samples} samples.")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.total_samples:
            raise IndexError("Index out of range")

        target_file_info = next(info for info in reversed(self.file_info) if idx >= info['start_index'])
        local_idx = idx - target_file_info['start_index']

        with h5py.File(target_file_info['path'], 'r') as f:
            image = f['images'][local_idx]
            command = f['commands'][local_idx]

        sample = {
            'image': image, 
            'command': torch.from_numpy(command.astype(np.float32))
        }

        if self.transform:
            sample = self.transform(sample)

        return sample