import numpy as np
import pandas as pd
import h5py
import hdf5plugin
from rosbags.highlevel import AnyReader
from cv_bridge import CvBridge
import cv2  # OpenCVをインポート
import os
import argparse
from pathlib import Path
from typing import Optional, Tuple

# --- トピック名は設定項目として残す ---
IMAGE_TOPIC = '/sensing/camera/image_raw'
CONTROL_TOPIC = '/awsim/control_cmd'
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
    単一のrosbagからデータを抽出し、同期してHDF5ファイルに保存する関数
    
    Args:
        bag_path_str (str): 入力rosbagファイルのパス
        h5_path_str (str): 出力HDF5ファイルのパス
        resize_dim (Optional[Tuple[int, int]]): 画像のリサイズ先 (幅, 高さ)。Noneの場合はリサイズしない。
    """
    bag_path = Path(bag_path_str)
    if not bag_path.exists():
        print(f"Error: Bag path does not exist: {bag_path}")
        return

    image_data, control_data = [], []
    print(f"Reading rosbag file from: {bag_path}")
    if resize_dim:
        print(f"  -> Images will be resized to {resize_dim[0]}x{resize_dim[1]}")

    try:
        with AnyReader([bag_path]) as reader:
            bridge = CvBridge()
            connections = [c for c in reader.connections if c.topic in [IMAGE_TOPIC, CONTROL_TOPIC]]
            for connection, _, rawdata in reader.messages(connections=connections):
                msg = reader.deserialize(rawdata, connection.msgtype)
                if connection.topic == IMAGE_TOPIC:
                    cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                    
                    # ★変更点: リサイズ処理を追加
                    if resize_dim:
                        cv_image = cv2.resize(cv_image, resize_dim, interpolation=cv2.INTER_AREA)
                        
                    image_data.append({'timestamp': msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec, 'image': cv_image})
                elif connection.topic == CONTROL_TOPIC:
                    control_data.append({'timestamp': msg.stamp.sec * 1e9 + msg.stamp.nanosec, 'speed': msg.longitudinal.speed, 'steering': msg.lateral.steering_tire_angle})
    except Exception as e:
        print(f"  -> Failed to read bag file: {e}")
        return

    if not image_data or not control_data:
        print("  -> Could not find image or control data.")
        return
    print(f"  -> Data read complete. Images: {len(image_data)}, Controls: {len(control_data)}")

    img_df = pd.DataFrame(image_data)
    img_df['timestamp_dt'] = pd.to_datetime(img_df['timestamp'])
    ctrl_df = pd.DataFrame(control_data)
    ctrl_df['timestamp_dt'] = pd.to_datetime(ctrl_df['timestamp'])

    print("  -> Synchronizing data...")
    synced_df = pd.merge_asof(img_df.sort_values('timestamp_dt'), ctrl_df.sort_values('timestamp_dt'), on='timestamp_dt', direction='nearest').dropna().reset_index(drop=True)
    
    if synced_df.empty:
        print("  -> No data after synchronization. Skipping file.")
        return
        
    print(f"  -> Synchronization complete. Synced samples: {len(synced_df)}")

    print(f"  -> Saving data to {h5_path_str}...")
    with h5py.File(h5_path_str, 'w') as hf:
        num_samples, img_shape = len(synced_df), synced_df['image'].iloc[0].shape
        blosc_opts = get_blosc_opts(complevel=1, complib='blosc:zstd', shuffle='byte')
        
        dset_images = hf.create_dataset('images', (num_samples, *img_shape), dtype=np.uint8, **blosc_opts)
        dset_commands = hf.create_dataset('commands', (num_samples, 2), dtype=np.float32, **blosc_opts)

        # ★変更点: データをNumpy配列に一括変換
        print("  -> Converting data to numpy arrays for bulk writing...")
        all_images = np.stack(synced_df['image'].to_list())
        all_commands = synced_df[['speed', 'steering']].to_numpy(dtype=np.float32)

        # ★変更点: Numpy配列を一括で書き込み
        print("  -> Writing data to HDF5 at once...")
        dset_images[:] = all_images
        dset_commands[:] = all_commands

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
    # ★変更点: リサイズ用の引数を追加
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
            # ★変更点: リサイズ指定を関数に渡す
            extract_and_sync_data(str(bag_path), str(output_filepath), resize_dim=resize_dimension)

        print("\n--- All processing finished. ---")