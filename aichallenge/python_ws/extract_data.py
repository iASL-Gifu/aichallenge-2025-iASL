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
    with open(config_path, 'r') as f:
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
    scanとcontrol_cmdを同期して書き出すことに特化。
    """
    print(f"\nProcessing: {input_path}\n        -> To: {output_path}")
    
    # --- 設定ファイルからキー名やパラメータを取得 ---
    cfg_keys = config['data_keys']
    cfg_preproc = config['preprocessing']
    cfg_output = config['output']

    scan_dset_name = cfg_keys['scan_dataset']
    control_dset_name = cfg_keys['control_dataset']
    
    print(f"  [{input_path.name}] Loading raw data into memory...")
    dataframes = {}
    scan_attributes = {}
    try:
        with h5py.File(input_path, 'r') as f:
            # --- Scanデータの読み込み ---
            if scan_dset_name in f:
                scan_df = pd.DataFrame(f[scan_dset_name][:])
                scan_df[cfg_keys['time_columns']['timestamp']] = scan_df[cfg_keys['time_columns']['sec']].astype(np.int64) * 1_000_000_000 + scan_df[cfg_keys['time_columns']['nanosec']].astype(np.int64)
                
                for key, value in f[scan_dset_name].attrs.items():
                    scan_attributes[key] = value

                print(f"  [{input_path.name}] Cleaning nan/inf in scan data...")
                range_max = scan_attributes.get('range_max', cfg_preproc['cleaning']['default_range_max'])
                
                ranges_col = cfg_keys['scan_columns']['ranges']
                intensities_col = cfg_keys['scan_columns']['intensities']
                
                scan_df[ranges_col] = scan_df[ranges_col].apply(
                    lambda r: np.nan_to_num(r, 
                                            nan=cfg_preproc['cleaning']['replace_nan_with'], 
                                            posinf=range_max, 
                                            neginf=cfg_preproc['cleaning']['replace_neginf_with']).astype(np.float32)
                )
                
                if intensities_col in scan_df.columns and scan_df[intensities_col].iloc[0] is not None:
                     scan_df[intensities_col] = scan_df[intensities_col].apply(
                        lambda i: np.nan_to_num(i, 
                                                nan=cfg_preproc['cleaning']['replace_nan_with'], 
                                                posinf=cfg_preproc['cleaning']['replace_posinf_with'], 
                                                neginf=cfg_preproc['cleaning']['replace_neginf_with']).astype(np.float32)
                    )
                dataframes[scan_dset_name] = scan_df
            else:
                print(f"  -> Error in {input_path.name}: '{scan_dset_name}' dataset not found. Cannot proceed.")
                return

            # --- Control Commandデータの読み込み ---
            if control_dset_name in f:
                control_df = pd.DataFrame(f[control_dset_name][:])
                control_df[cfg_keys['time_columns']['timestamp']] = control_df[cfg_keys['time_columns']['sec']].astype(np.int64) * 1_000_000_000 + control_df[cfg_keys['time_columns']['nanosec']].astype(np.int64)
                dataframes[control_dset_name] = control_df
            else:
                print(f"  -> Error in {input_path.name}: '{control_dset_name}' dataset not found. Cannot proceed.")
                return

    except Exception as e:
        print(f"  -> Failed to read raw HDF5 file {input_path.name}: {e}")
        return

    print(f"  [{input_path.name}] Synchronizing scan and control data...")
    if scan_dset_name not in dataframes or control_dset_name not in dataframes:
        print(f"  -> Missing required data in {input_path.name}. Skipping.")
        return
        
    merged_df = dataframes.pop(scan_dset_name).sort_values(cfg_keys['time_columns']['timestamp'])
    control_df = dataframes.pop(control_dset_name)
    
    tolerance_ns = int(cfg_preproc['sync']['tolerance_seconds'] * 1_000_000_000)
    
    merged_df = pd.merge_asof(merged_df, control_df.sort_values(cfg_keys['time_columns']['timestamp']), 
                              on=cfg_keys['time_columns']['timestamp'], direction='nearest', tolerance=tolerance_ns, 
                              suffixes=('', cfg_preproc['sync']['merge_suffix']))
    
    merged_df.dropna(inplace=True)
    merged_df.reset_index(drop=True, inplace=True)
    
    # === フィルタリング処理を削除 ===
    
    if merged_df.empty:
        print(f"  -> No data after synchronization in {input_path.name}. Skipping file.")
        return
        
    num_samples = len(merged_df)
    
    print(f"  [{input_path.name}] Writing {num_samples} samples to new HDF5 file...")
    with h5py.File(output_path, 'w') as f_out:
        vlen_float_dtype = h5py.vlen_dtype(np.float32)
        scan_dtype = np.dtype([
            (cfg_keys['time_columns']['sec'], 'i4'), (cfg_keys['time_columns']['nanosec'], 'u4'),
            (cfg_keys['scan_columns']['ranges'], vlen_float_dtype),
            (cfg_keys['scan_columns']['intensities'], vlen_float_dtype)
        ])
        dset_scan = f_out.create_dataset(scan_dset_name, (num_samples,), dtype=scan_dtype, **hdf5plugin.Blosc())
        
        for key, value in scan_attributes.items():
            dset_scan.attrs[key] = value
        
        control_columns_to_write = cfg_output['control_columns_to_write']
        dset_control = f_out.create_dataset(control_dset_name, (num_samples, len(control_columns_to_write)), dtype=np.float32, **hdf5plugin.Blosc())

        scan_data_to_write = np.empty((num_samples,), dtype=scan_dtype)
        scan_data_to_write[cfg_keys['time_columns']['sec']] = merged_df[cfg_keys['time_columns']['sec']].to_numpy(dtype='i4')
        scan_data_to_write[cfg_keys['time_columns']['nanosec']] = merged_df[cfg_keys['time_columns']['nanosec']].to_numpy(dtype='u4')
        scan_data_to_write[cfg_keys['scan_columns']['ranges']] = merged_df[cfg_keys['scan_columns']['ranges']].to_list()
        scan_data_to_write[cfg_keys['scan_columns']['intensities']] = merged_df[cfg_keys['scan_columns']['intensities']].to_list()
        
        dset_scan[:] = scan_data_to_write
        
        control_data_to_write = merged_df[control_columns_to_write].to_numpy(dtype=np.float32)
        dset_control[:] = control_data_to_write
                
    print(f"  -> Done processing {input_path.name}.")


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description='Preprocess raw HDF5 files focusing on Scan and Control data.')
    parser.add_argument('input_dir', type=str, help='Directory containing raw HDF5 files.')
    parser.add_argument('output_dir', type=str, help='Directory to save the preprocessed HDF5 files.')
    parser.add_argument('--config', type=str, default='config/preprocess.yaml', help='Path to the preprocess config file.')
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