import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import cv2
from typing import  Dict, Any, Callable

class HDF5Dataset(Dataset):
    """
    HDF5ファイルを読み込むためのDatasetクラス。
    """
    def __init__(self, h5_path: str, transform: Callable = None):
        self.h5_path = h5_path
        self.transform = transform
        self._archives: Dict[int, Any] = {}
        with h5py.File(self.h5_path, 'r') as hf:
            self.length = hf['images'].shape[0]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Dict[str, Any]:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else -1
        if worker_id not in self._archives:
             self._archives[worker_id] = h5py.File(self.h5_path, 'r')
        h5_file = self._archives[worker_id]
        
        start_idx, end_idx = h5_file['frame_indices'][index]

        image = cv2.cvtColor(h5_file['images'][index], cv2.COLOR_BGR2RGB)
        command = h5_file['commands'][index].astype(np.float32)
        
        # HDF5ファイルから直接、分離されたデータを読み込む
        timeseries_control = h5_file['timeseries_control'][start_idx:end_idx].astype(np.float32)
        timeseries_imu = h5_file['timeseries_imu'][start_idx:end_idx].astype(np.float32)

        output = {
            'image': image,
            'command': command,
            'timeseries_control': timeseries_control,
            'timeseries_imu': timeseries_imu,
        }

        if self.transform:
            output = self.transform(output)

        return output