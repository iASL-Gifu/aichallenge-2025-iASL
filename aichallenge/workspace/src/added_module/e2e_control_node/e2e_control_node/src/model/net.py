import torch
import torch.nn as nn
import torchvision.models as models

class DrivingModel(nn.Module):
    def __init__(self, model_name='resnet18', pretrained=True):
        """
        Args:
            model_name (str): 使用するResNetのバージョン ('resnet18', 'resnet34', etc.)
            pretrained (bool): 事前学習済み重みを使用するかどうか
        """
        super().__init__()
        
        # 1. 事前学習済みのResNetをベースとしてロード
        if pretrained:
            weights = models.get_model_weights(model_name).IMAGENET1K_V1
            self.resnet_base = models.get_model(model_name, weights=weights)
        else:
            self.resnet_base = models.get_model(model_name, weights=None)

        # 2. 最終の全結合層（分類用）を特徴抽出のためのIdentityに置き換え
        num_features = self.resnet_base.fc.in_features
        self.resnet_base.fc = nn.Identity()

        # 3. 2つの出力ヘッドを定義
        # アクセル用ヘッド (出力: 1)
        self.accel_head = nn.Linear(num_features, 1)
        # ステアリング用ヘッド (出力: 1)
        self.steer_head = nn.Linear(num_features, 1)

    def forward(self, x):
        """
        順伝播の定義
        """
        # ResNetで画像から特徴量を抽出
        features = self.resnet_base(x)
        
        # 各ヘッドで値を推論
        raw_accel = self.accel_head(features)
        raw_steer = self.steer_head(features)
        
        # 活性化関数で出力をスケーリング
        accel = torch.sigmoid(raw_accel)  # 0〜1の範囲に
        steer = torch.tanh(raw_steer)    # -1〜1の範囲に
        
        # 辞書形式で出力を返す
        return {
            'accel': accel.squeeze(-1), # [batch, 1] -> [batch]
            'steer': steer.squeeze(-1)  # [batch, 1] -> [batch]
        }