import numpy as np
import pandas as pd
import h5py
import hdf5plugin
from rosbags.highlevel import AnyReader
from cv_bridge import CvBridge
import os
import argparse
from pathlib import Path

# --- トピック名は設定項目として残す ---
IMAGE_TOPIC = '/sensing/camera/image_raw'
CONTROL_TOPIC = '/awsim/control_cmd'
# ----------------------------------

def extract_and_sync_data(bag_path_str: str, h5_path_str: str):
    """
    【この関数は変更なし】
    単一のrosbagからデータを抽出し、同期してHDF5ファイルに保存する関数
    """
    bag_path = Path(bag_path_str)
    if not bag_path.exists():
        print(f"Error: Bag path does not exist: {bag_path}")
        return

    image_data, control_data = [], []
    print(f"Reading rosbag file from: {bag_path}")
    try:
        with AnyReader([bag_path]) as reader:
            bridge = CvBridge()
            connections = [c for c in reader.connections if c.topic in [IMAGE_TOPIC, CONTROL_TOPIC]]
            for connection, _, rawdata in reader.messages(connections=connections):
                msg = reader.deserialize(rawdata, connection.msgtype)
                if connection.topic == IMAGE_TOPIC:
                    image_data.append({'timestamp': msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec, 'image': bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')})
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
    print(f"  -> Synchronization complete. Synced samples: {len(synced_df)}")

    print(f"  -> Saving data to {h5_path_str}...")
    with h5py.File(h5_path_str, 'w') as hf:
        num_samples, img_shape = len(synced_df), synced_df['image'].iloc[0].shape
        blosc_opts = hdf5plugin.Blosc(cname='lz4', clevel=1, shuffle=hdf5plugin.Blosc.SHUFFLE)
        dset_images = hf.create_dataset('images', (num_samples, *img_shape), dtype=np.uint8, **blosc_opts)
        dset_commands = hf.create_dataset('commands', (num_samples, 2), dtype=np.float32, **blosc_opts)
        for i, row in synced_df.iterrows():
            dset_images[i] = row['image']
            dset_commands[i] = [row['speed'], row['steering']]
    print("  -> Done.")

def find_rosbag_directories(root_search_path: Path) -> list[Path]:
    """
    ★新規追加★
    指定されたパス以下を再帰的に探索し、rosbagのディレクトリ（metadata.yamlを含む）のリストを返す関数
    """
    bag_directories = []
    print(f"Searching for rosbags in '{root_search_path}'...")
    for dirpath, _, filenames in os.walk(root_search_path):
        if 'metadata.yaml' in filenames:
            bag_path = Path(dirpath)
            bag_directories.append(bag_path)
            print(f"  -> Found: {bag_path}")
    return bag_directories

if __name__ == '__main__':
    # ★変更点: コマンドライン引数の定義を更新
    parser = argparse.ArgumentParser(description='Find all rosbags in a directory and process them into HDF5 files.')
    parser.add_argument('search_dir', type=str, help='Root directory to search for rosbags.')
    parser.add_argument('output_dir', type=str, help='Directory to save the output HDF5 files.')
    args = parser.parse_args()

    root_path = Path(args.search_dir)
    output_root_path = Path(args.output_dir)

    # 出力ディレクトリが存在しない場合は作成
    output_root_path.mkdir(parents=True, exist_ok=True)

    # rosbagディレクトリを全て見つける
    rosbag_paths = find_rosbag_directories(root_path)

    if not rosbag_paths:
        print("No rosbags were found.")
    else:
        print(f"\n--- Found {len(rosbag_paths)} rosbag(s). Starting processing. ---")
        # 見つかった各rosbagに対して処理を実行
        for i, bag_path in enumerate(rosbag_paths):
            # rosbagのフォルダ名から出力ファイル名を生成
            output_filename = bag_path.name + ".h5"
            output_filepath = output_root_path / output_filename
            
            print(f"\n[{i+1}/{len(rosbag_paths)}] Processing '{bag_path.name}' -> '{output_filepath}'")
            extract_and_sync_data(str(bag_path), str(output_filepath))

        print("\n--- All processing finished. ---")