import os
import h5py
import hdf5plugin
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
import argparse
from tqdm import tqdm
import multiprocessing
from functools import partial

def load_config(config_path):
    """YAML設定ファイルを読み込むヘルパー関数"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def preprocess_h5(h5_path, input_root, output_root, config):
    """
    ワーカープロセスのためのラッパー関数。
    """
    relative_path = h5_path.relative_to(input_root)
    output_h5_path = output_root / relative_path
    output_h5_path.parent.mkdir(parents=True, exist_ok=True)
    
    _preprocess_h5_internal(h5_path, output_h5_path, config)

def _preprocess_h5_internal(input_path, output_path, config):
    """
    単一のHDF5ファイルに対する前処理のコアロジック。
    YAML設定に基づいて動的にフィルタリングを行う。
    """
    print(f"\nProcessing: {input_path}\n        -> To: {output_path}")
    
    # --- 設定ファイルからキー名やパラメータを取得 ---
    cfg_keys = config.get('data_keys', {})
    cfg_filter = config.get('filtering', {})
    cfg_output = config.get('output', {})

    scan_dset_name = cfg_keys.get('scan_dataset', 'scan')
    control_dset_name = cfg_keys.get('control_dataset', 'control_cmd')
    
    print(f"  [{input_path.name}] Loading pre-processed data into memory...")
    try:
        # extractで整形済みのデータを読み込むことを想定
        with h5py.File(input_path, 'r') as f:
            if scan_dset_name not in f or control_dset_name not in f:
                print(f"  -> Error in {input_path.name}: Required dataset not found. Skipping.")
                return

            scan_df = pd.DataFrame(f[scan_dset_name][:])
            control_df = pd.DataFrame(f[control_dset_name][:])
            scan_attributes = {key: value for key, value in f[scan_dset_name].attrs.items()}
    
    except Exception as e:
        print(f"  -> Failed to read HDF5 file {input_path.name}: {e}")
        return
        
    # extractで同期済みのデータを読み込むため、ここでは単純なマージを行う
    # もしextractで同期していない場合は、以前のmerge_asofロジックをここに記述
    timestamp_col = cfg_keys.get('time_columns', {}).get('timestamp', 'timestamp')
    if timestamp_col not in scan_df.columns:
         scan_df[timestamp_col] = scan_df[cfg_keys.get('time_columns', {}).get('sec', 'sec')].astype(np.int64) * 1_000_000_000 + scan_df[cfg_keys.get('time_columns', {}).get('nanosec', 'nanosec')].astype(np.int64)
         control_df[timestamp_col] = control_df[cfg_keys.get('time_columns', {}).get('sec', 'sec')].astype(np.int64) * 1_000_000_000 + control_df[cfg_keys.get('time_columns', {}).get('nanosec', 'nanosec')].astype(np.int64)

    merged_df = pd.merge(scan_df, control_df, on=timestamp_col, how='inner')

    # === フィルタリング Step 1: 人間介入データ ===
    if cfg_filter.get('human_intervention', {}).get('enabled', False):
        filter_cfg = cfg_filter['human_intervention']
        column = filter_cfg['column']
        flag_value = filter_cfg['flag_value']
        
        print(f"  [{input_path.name}] Filtering for human intervention data ({column} == {flag_value})...")
        original_count = len(merged_df)
        if column in merged_df.columns:
            merged_df = merged_df[np.isclose(merged_df[column], flag_value)].copy()
            merged_df.reset_index(drop=True, inplace=True)
            filtered_count = len(merged_df)
            print(f"  -> Filtered samples by intervention: {original_count} -> {filtered_count}")
        else:
            print(f"  -> Warning: '{column}' column not found. Cannot filter. Keeping all {original_count} samples.")

    # === フィルタリング Step 2: 指定されたラップ数 ===
    if cfg_filter.get('target_laps', {}).get('enabled', False):
        filter_cfg = cfg_filter['target_laps']
        target_laps = filter_cfg.get('laps')
        column = filter_cfg['column']
        
        if target_laps and isinstance(target_laps, list):
            print(f"  [{input_path.name}] Filtering for target laps: {target_laps} using column '{column}'...")
            original_count_lap = len(merged_df)
            
            if column in merged_df.columns:
                merged_df = merged_df[merged_df[column].round().astype(int).isin(target_laps)].copy()
                merged_df.reset_index(drop=True, inplace=True)
                filtered_count_lap = len(merged_df)
                print(f"  -> Filtered samples by lap: {original_count_lap} -> {filtered_count_lap}")
            else:
                 print(f"  -> Warning: '{column}' column for laps not found. Cannot filter.")

    if merged_df.empty:
        print(f"  -> No data after filtering in {input_path.name}. Skipping file.")
        return
        
    num_samples = len(merged_df)
    
    print(f"  [{input_path.name}] Writing {num_samples} filtered samples to new HDF5 file...")
    with h5py.File(output_path, 'w') as f_out:
        # --- データセットの作成 ---
        vlen_float_dtype = h5py.vlen_dtype(np.float32)
        scan_dtype = np.dtype([
            (cfg_keys.get('time_columns', {}).get('sec', 'sec'), 'i4'), 
            (cfg_keys.get('time_columns', {}).get('nanosec', 'nanosec'), 'u4'),
            (cfg_keys.get('scan_columns', {}).get('ranges', 'ranges'), vlen_float_dtype),
            (cfg_keys.get('scan_columns', {}).get('intensities', 'intensities'), vlen_float_dtype)
        ])
        dset_scan = f_out.create_dataset(scan_dset_name, (num_samples,), dtype=scan_dtype, **hdf5plugin.Blosc())
        
        for key, value in scan_attributes.items():
            dset_scan.attrs[key] = value
        
        control_columns_to_write = cfg_output.get('control_columns_to_write', [])
        dset_control = f_out.create_dataset(control_dset_name, (num_samples, len(control_columns_to_write)), dtype=np.float32, **hdf5plugin.Blosc())

        # --- データの書き込み ---
        scan_data_to_write = np.empty((num_samples,), dtype=scan_dtype)
        for col_name, _ in scan_dtype.fields.items():
            if col_name in merged_df:
                scan_data_to_write[col_name] = merged_df[col_name].to_numpy() if 'vlen' not in str(scan_dtype[col_name]) else merged_df[col_name].to_list()
        
        dset_scan[:] = scan_data_to_write
        
        control_data_to_write = merged_df[control_columns_to_write].to_numpy(dtype=np.float32)
        dset_control[:] = control_data_to_write
                
    print(f"  -> Done processing {input_path.name}.")

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description='Preprocess HDF5 files by filtering based on a config file.')
    parser.add_argument('input_dir', type=str, help='Directory containing HDF5 files to be preprocessed.')
    parser.add_argument('output_dir', type=str, help='Directory to save the filtered HDF5 files.')
    parser.add_argument('--config', type=str, default='config/preprocess_data.yaml', help='Path to the preprocess config file.')
    args = parser.parse_args()

    input_root, output_root, config_path = Path(args.input_dir), Path(args.output_dir), Path(args.config)

    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        exit(1)

    config = load_config(config_path)
    output_root.mkdir(parents=True, exist_ok=True)

    h5_files = sorted(list(input_root.glob('**/*.h5')))
    if not h5_files:
        print(f"No .h5 files found recursively in {input_root}")
    else:
        print(f"Found {len(h5_files)} files to process.")
        
        num_jobs = config.get('num_workers', multiprocessing.cpu_count()) 
        print(f"Starting parallel processing with {num_jobs} jobs...")

        worker_func = partial(preprocess_h5, input_root=input_root, output_root=output_root, config=config)
        
        with multiprocessing.Pool(processes=num_jobs) as pool:
            list(tqdm(pool.imap_unordered(worker_func, h5_files), total=len(h5_files), desc="Overall Progress"))
            
        print("\n--- All preprocessing finished. ---")