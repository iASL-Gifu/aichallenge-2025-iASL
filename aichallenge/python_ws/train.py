import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import argparse
import yaml  
import os

from src.data.dataset import DrivingDataset
from src.data.transform import get_transforms
from src.model.net import DrivingModel
from src.model.ema import ModelEMA

def train_one_epoch(model, ema_model, dataloader, criterion, optimizer, device, scaler):
    """1エポック分の学習を行う関数 (この関数の中身は変更なし)"""
    model.train()
    running_loss = 0.0
    for i, batch in enumerate(dataloader):
        images = batch['image'].to(device)
        true_commands = batch['command'].to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            outputs = model(images)
            loss_accel = criterion(outputs['accel'], true_commands[:, 0])
            loss_steer = criterion(outputs['steer'], true_commands[:, 1])
            total_loss = loss_accel + loss_steer
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        ema_model.update(model)
        running_loss += total_loss.item()
    return running_loss / len(dataloader)

def main(config):
    """メインの学習処理"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loaded config: {config}")

    # 1. TransformとDataLoaderの準備
    transforms = get_transforms()
    train_dataset = DrivingDataset(data_dir=config['data_dir'], transform=transforms['train'])
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True, 
        num_workers=config['dataloader']['num_workers'],
        pin_memory=True
    )

    # 2. モデル、損失関数、最適化手法の定義
    model = DrivingModel(model_name=config['model']['name']).to(device)
    ema_model = ModelEMA(model, decay=config['model']['ema_decay'])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    scaler = GradScaler()

    # 3. 学習ループの実行
    print("\n--- Starting Training ---")
    for epoch in range(config['training']['epochs']):
        epoch_loss = train_one_epoch(model, ema_model, train_dataloader, criterion, optimizer, device, scaler)
        print(f"\nEpoch {epoch+1}/{config['training']['epochs']} finished. Average Loss: {epoch_loss:.4f}")

        # モデルの保存
        save_dir = config['save_dir']
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'ema_model_epoch_{epoch+1}.pth')
            torch.save(ema_model.ema_model.state_dict(), save_path)
            print(f"Saved EMA model to {save_path}")

    print("\n--- Training Finished ---")

if __name__ == '__main__':
    # ★ argparseでYAMLファイルのパスのみを受け取る
    parser = argparse.ArgumentParser(description='Train a driving model using a config file.')
    parser.add_argument(
        '--config', 
        type=str, 
        required=True, # 必須引数にする
        help='Path to the YAML configuration file.'
    )
    args = parser.parse_args()

    # YAMLファイルを読み込む
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # メイン関数を実行
    main(config)