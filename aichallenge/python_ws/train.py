import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse

from src.data.dataset import DrivingDataset
from src.data.transform import get_transforms
from src.model.net import DrivingModel

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """1エポック分の学習を行う関数"""
    model.train()  # モデルを学習モードに
    running_loss = 0.0

    for i, batch in enumerate(dataloader):
        # データをデバイスに転送
        images = batch['image'].to(device)
        true_commands = batch['command'].to(device)
        
        # 勾配をリセット
        optimizer.zero_grad()
        
        # 順伝播
        outputs = model(images)
        
        # 損失の計算
        # true_commands[:, 0] は速度、 [:, 1] はステアリングと仮定
        loss_accel = criterion(outputs['accel'], true_commands[:, 0])
        loss_steer = criterion(outputs['steer'], true_commands[:, 1])
        total_loss = loss_accel + loss_steer  # 2つの損失を合算

        # 逆伝播
        total_loss.backward()
        
        # パラメータの更新
        optimizer.step()
        
        running_loss += total_loss.item()
        if (i + 1) % 50 == 0:  # 50バッチごとに進捗を表示
            print(f"  Batch {i+1}/{len(dataloader)}, Loss: {total_loss.item():.4f}")

    epoch_loss = running_loss / len(dataloader)
    return epoch_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a driving model.')
    parser.add_argument('--data_dir', type=str, default='./output_datasets', help='Path to the dataset directory.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    args = parser.parse_args()

    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. TransformとDataLoaderの準備
    transforms = get_transforms()
    train_dataset = DrivingDataset(data_dir=args.data_dir, transform=transforms['train'])
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    # 2. モデル、損失関数、最適化手法の定義
    model = DrivingModel().to(device)
    criterion = nn.MSELoss() # 平均二乗誤差
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 3. 学習ループの実行
    print("\n--- Starting Training ---")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        epoch_loss = train_one_epoch(model, train_dataloader, criterion, optimizer, device)
        print(f"Epoch {epoch+1} finished. Average Loss: {epoch_loss:.4f}")
        # ここにモデルの保存処理などを追加
        torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')

    print("\n--- Training Finished ---")