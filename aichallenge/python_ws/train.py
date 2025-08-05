import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
from tqdm import tqdm
import numpy as np
import yaml

from src.model.dual_net import ModelA_StateFixed, ModelB_StateUpdating, ModelC_HybridSkipConnection
from src.data.dataset import HDF5Dataset
from src.data.collate import custom_collate_fn
from src.data.transform import train_transform
from src.model.ema import ModelEMA 

# --- 利用可能なモデルを辞書にマッピング ---
MODEL_MAP = {
    "ModelA_StateFixed": ModelA_StateFixed,
    "ModelB_StateUpdating": ModelB_StateUpdating,
    "ModelC_HybridSkipConnection": ModelC_HybridSkipConnection,
}

def train_one_epoch(model, ema, dataloader, criterion, optimizer, device, lengths_key='timeseries_lengths'):
    """1エポック分の学習を行う関数"""
    model.train()
    running_loss = 0.0
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        images = batch['image'].to(device)
        imu_sequences = batch['timeseries_imu'].to(device)
        target_controls = batch['timeseries_control'].to(device)
        lengths = batch[lengths_key].to(device)
        
        optimizer.zero_grad()
        predicted_controls = model(images, imu_sequences)
        
        max_len = predicted_controls.shape[1]
        target_controls = target_controls[:, :max_len, :]
        mask = torch.arange(max_len, device=device)[None, :] < lengths[:, None]
        loss = criterion(predicted_controls, target_controls)
        masked_loss = (loss * mask.unsqueeze(-1)).sum() / mask.sum()
        
        masked_loss.backward()
        optimizer.step()
        
        if ema:
            ema.update(model)

        running_loss += masked_loss.item()
        pbar.set_postfix(loss=masked_loss.item())
        
    return running_loss / len(dataloader)

def main(config):
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = HDF5Dataset(h5_path=config['data_path'], transform=train_transform)
    print(f"Training set size: {len(train_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=custom_collate_fn,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )

    model_name = config['model_name']
    model_class = MODEL_MAP.get(model_name)
    model = model_class(**config.get('model_params', {})).to(device)
    print(f"Model: {model_name} loaded.")
    
    ema_cfg = config.get('ema', {})
    if ema_cfg.get('use', False):
        ema = ModelEMA(model, decay=ema_cfg.get('decay', 0.999))
        print(f"Using ModelEMA with decay {ema.decay}")
    else:
        ema = None

    criterion = nn.MSELoss(reduction='none')
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])
    
    scheduler_cfg = config.get('scheduler', {})
    scheduler = None
    if scheduler_cfg.get('type') == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                            step_size=scheduler_cfg.get('step_size', 30), 
                                            gamma=scheduler_cfg.get('gamma', 0.1))
        print(f"Using StepLR scheduler with step_size={scheduler.step_size} and gamma={scheduler.gamma}")
    else:
        print("No scheduler used.")

    best_train_loss = np.inf
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(config['training']['epochs']):
        train_loss = train_one_epoch(model, ema, train_loader, criterion, optimizer, device)
        
        if scheduler:
            scheduler.step()
        
        print(f"Epoch {epoch+1}/{config['training']['epochs']} -> Train Loss: {train_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            # 保存するのはEMAモデル（使用している場合）
            model_to_save = ema.ema_model if ema else model
            model_path = os.path.join(output_dir, f"best_model_{model_name}.pth")
            torch.save(model_to_save.state_dict(), model_path)
            print(f"-> Best model saved to {model_path} (Train Loss: {best_train_loss:.6f})")

    # 学習終了後、最終エポックのモデルも保存する
    final_model_to_save = ema.ema_model if ema else model
    final_model_path = os.path.join(output_dir, f"final_model_{model_name}_epoch{config['training']['epochs']}.pth")
    torch.save(final_model_to_save.state_dict(), final_model_path)
    print(f"Training finished. Final model saved to {final_model_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a driving model using a config file.')
    parser.add_argument(
        '--config', 
        type=str, 
        required=True,
        help='Path to the YAML configuration file.'
    )
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    main(config)