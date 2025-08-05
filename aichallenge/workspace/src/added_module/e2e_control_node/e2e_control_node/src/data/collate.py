import torch
from typing import Dict, Any, List    

def custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    長さが異なる時系列データをパディングしてバッチを作成する関数。
    """
    images = torch.stack([item['image'] for item in batch], dim=0)
    commands = torch.stack([item['command'] for item in batch], dim=0)
    timeseries_control_list = [item['timeseries_control'] for item in batch]
    timeseries_imu_list = [item['timeseries_imu'] for item in batch]
    
    lengths = torch.tensor([len(ts) for ts in timeseries_control_list], dtype=torch.long)
    
    padded_control = torch.nn.utils.rnn.pad_sequence(
        timeseries_control_list, batch_first=True, padding_value=0.0
    )
    padded_imu = torch.nn.utils.rnn.pad_sequence(
        timeseries_imu_list, batch_first=True, padding_value=0.0
    )
    output = {
        'image': images,
        'command': commands,
        'timeseries_control': padded_control,
        'timeseries_imu': padded_imu,
        'timeseries_lengths': lengths
    }
    return output