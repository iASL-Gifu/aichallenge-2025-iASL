import numpy as np
import pandas as pd
import h5py
import hdf5plugin
from rosbags.highlevel import AnyReader
from cv_bridge import CvBridge
import cv2  
import os
import argparse
from pathlib import Path
from typing import Optional, Tuple

# --- トピック名は設定項目として残す ---
IMAGE_TOPIC = '/sensing/camera/image_raw'
CONTROL_TOPIC = '/awsim/control_cmd'
IMU_TOPIC = '/sensing/imu/imu_raw'
# ----------------------------------

def get_blosc_opts(complevel=1, complib='blosc:zstd', shuffle='byte'):
    """Blosc圧縮の設定を返すヘルパー関数"""
    shuffle_map = {'bit': 2, 'byte': 1, 'none': 0}
    shuffle_val = shuffle_map.get(shuffle, 0)
    compressors = ['blosclz', 'lz4', 'lz4hc', 'snappy', 'zlib', 'zstd']
    complib_val = ['blosc:' + c for c in compressors].index(complib)
    args = {
        'compression': 32001,
        'compression_opts': (0, 0, 0, 0, complevel, shuffle_val, complib_val),
        'chunks': True
    }
    return args

def extract_and_sync_data(bag_path_str: str, h5_path_str: str, resize_dim: Optional[Tuple[int, int]] = None):
    """
    rosbagからデータを抽出し、画像を基準として高周波データを紐付けてHDF5ファイルに保存する関数
    """
    bag_path = Path(bag_path_str)
    if not bag_path.exists():
        print(f"Error: Bag path does not exist: {bag_path}")
        return

    image_data, control_data, imu_data = [], [], []
    print(f"Reading rosbag file from: {bag_path}")

    try:
        with AnyReader([bag_path]) as reader:
            bridge = CvBridge()
            connections = [c for c in reader.connections if c.topic in [IMAGE_TOPIC, CONTROL_TOPIC, IMU_TOPIC]]
            for connection, _, rawdata in reader.messages(connections=connections):
                msg = reader.deserialize(rawdata, connection.msgtype)
                
                if connection.topic == IMAGE_TOPIC:
                    timestamp_ns = msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec
                    cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                    if resize_dim:
                        cv_image = cv2.resize(cv_image, resize_dim, interpolation=cv2.INTER_AREA)
                    image_data.append({'timestamp': timestamp_ns, 'image': cv_image})
                    
                elif connection.topic == CONTROL_TOPIC:
                    timestamp_ns = msg.stamp.sec * 1e9 + msg.stamp.nanosec
                    control_data.append({'timestamp': timestamp_ns, 'speed': msg.longitudinal.speed, 'steering': msg.lateral.steering_tire_angle})
                
                elif connection.topic == IMU_TOPIC:
                    timestamp_ns = msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec
                    imu_data.append({
                        'timestamp': timestamp_ns,
                        'angular_velocity_x': msg.angular_velocity.x, 'angular_velocity_y': msg.angular_velocity.y, 'angular_velocity_z': msg.angular_velocity.z,
                        'linear_acceleration_x': msg.linear_acceleration.x, 'linear_acceleration_y': msg.linear_acceleration.y, 'linear_acceleration_z': msg.linear_acceleration.z,
                    })
    except Exception as e:
        print(f"  -> Failed to read bag file: {e}")
        return

    if not image_data or not control_data or not imu_data:
        print("  -> Could not find image, control or imu data.")
        return
    print(f"  -> Data read complete. Images: {len(image_data)}, Controls: {len(control_data)}, IMU: {len(imu_data)}")

    # DataFrameに変換し、タイムスタンプでソート
    img_df = pd.DataFrame(image_data).sort_values('timestamp').reset_index(drop=True)
    ctrl_df = pd.DataFrame(control_data).sort_values('timestamp').reset_index(drop=True)
    imu_df = pd.DataFrame(imu_data).sort_values('timestamp').reset_index(drop=True)

    # 高周波データ(IMU, 制御)を結合・整理
    print("  -> Merging and sorting high-frequency data (IMU + Control)...")
    ctrl_df_ts = ctrl_df.set_index('timestamp')
    imu_df_ts = imu_df.set_index('timestamp')
    hf_df = pd.concat([ctrl_df_ts, imu_df_ts], axis=1).sort_index().interpolate(method='time')
    hf_df = hf_df.dropna().reset_index()
    
    if hf_df.empty:
        print("  -> No high-frequency data after merging. Skipping file.")
        return
        
    # 結合後のDataFrameから、制御とIMUのデータを別々のNumpy配列として抽出
    hf_timestamps = hf_df['timestamp'].to_numpy()
    control_columns = ['speed', 'steering']
    imu_columns = [
        'angular_velocity_x', 'angular_velocity_y', 'angular_velocity_z',
        'linear_acceleration_x', 'linear_acceleration_y', 'linear_acceleration_z'
    ]
    hf_control_values = hf_df[control_columns].to_numpy(dtype=np.float32)
    hf_imu_values = hf_df[imu_columns].to_numpy(dtype=np.float32)

    # 画像を基準に、高周波データのインデックスを作成
    print("  -> Creating frame indices for high-frequency data...")
    img_timestamps = img_df['timestamp'].to_numpy()
    indices = np.searchsorted(hf_timestamps, img_timestamps, side='right')
    
    frame_indices = []
    valid_images = []
    for i in range(len(indices) - 1):
        start_idx, end_idx = indices[i], indices[i+1]
        if start_idx < end_idx:
            frame_indices.append([start_idx, end_idx])
            valid_images.append(i)
    
    if not frame_indices:
        print("  -> Could not find any valid frames with corresponding high-frequency data. Skipping file.")
        return
        
    img_df_valid = img_df.iloc[valid_images].reset_index(drop=True)
    final_df = pd.merge_asof(img_df_valid, ctrl_df, on='timestamp', direction='backward')

    print(f"  -> Synchronization complete. Valid image frames: {len(final_df)}")

    # HDF5ファイルに保存
    print(f"  -> Saving data to {h5_path_str}...")
    with h5py.File(h5_path_str, 'w') as hf:
        blosc_opts = get_blosc_opts()
        num_samples = len(final_df)
        img_shape = final_df['image'].iloc[0].shape

        hf.create_dataset('images', (num_samples, *img_shape), dtype=np.uint8, **blosc_opts)
        hf.create_dataset('commands', (num_samples, 2), dtype=np.float32, **blosc_opts)
        hf.create_dataset('timeseries_control', data=hf_control_values, dtype=np.float32, **blosc_opts)
        hf.create_dataset('timeseries_imu', data=hf_imu_values, dtype=np.float32, **blosc_opts)
        hf.create_dataset('timeseries_timestamps', data=hf_timestamps, dtype=np.int64, **blosc_opts)
        hf.create_dataset('frame_indices', data=np.array(frame_indices, dtype=np.int64), dtype=np.int64, **blosc_opts)

        hf['images'][:] = np.stack(final_df['image'].to_list())
        hf['commands'][:] = final_df[['speed', 'steering']].to_numpy(dtype=np.float32)

    print("  -> Done.")

def find_rosbag_directories(root_search_path: Path) -> list[Path]:
    """指定されたパス以下を再帰的に探索し、rosbagのディレクトリのリストを返す関数"""
    bag_directories = []
    print(f"Searching for rosbags in '{root_search_path}'...")
    for dirpath, _, filenames in os.walk(root_search_path):
        if 'metadata.yaml' in filenames:
            bag_path = Path(dirpath)
            bag_directories.append(bag_path)
            print(f"  -> Found: {bag_path}")
    return bag_directories

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find all rosbags in a directory and process them into HDF5 files.')
    parser.add_argument('search_dir', type=str, help='Root directory to search for rosbags.')
    parser.add_argument('output_dir', type=str, help='Directory to save the output HDF5 files.')
    parser.add_argument('--resize', type=int, nargs=2, metavar=('WIDTH', 'HEIGHT'), 
                        help='Resize images to a specific dimension (e.g., --resize 640 360).')

    args = parser.parse_args()
    root_path = Path(args.search_dir)
    output_root_path = Path(args.output_dir)
    resize_dimension = tuple(args.resize) if args.resize else None
    output_root_path.mkdir(parents=True, exist_ok=True)
    rosbag_paths = find_rosbag_directories(root_path)

    if not rosbag_paths:
        print("No rosbags were found.")
    else:
        print(f"\n--- Found {len(rosbag_paths)} rosbag(s). Starting processing. ---")
        for i, bag_path in enumerate(rosbag_paths):
            output_filename = bag_path.name + ".h5"
            output_filepath = output_root_path / output_filename
            print(f"\n[{i+1}/{len(rosbag_paths)}] Processing '{bag_path.name}' -> '{output_filepath}'")
            extract_and_sync_data(str(bag_path), str(output_filepath), resize_dim=resize_dimension)
        print("\n--- All processing finished. ---")