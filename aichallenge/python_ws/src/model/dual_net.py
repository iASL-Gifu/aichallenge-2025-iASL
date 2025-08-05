import torch
import torch.nn as nn
import torchvision.models as models

class ModelA_StateFixed(nn.Module):
    def __init__(self, vision_feature_dim=512, imu_plan_hidden_dim=128, vision_plan_dim=64, dropout_p=0.2):
        super().__init__()
        
        # --- 低頻度ブランチ（プランナー）の定義 ---
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.vision_encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.vision_fc = nn.Linear(resnet.fc.in_features, vision_feature_dim)
        self.imu_plan_encoder = nn.LSTM(input_size=6, hidden_size=imu_plan_hidden_dim, num_layers=2, batch_first=True, dropout=dropout_p)
        self.fusion_fc = nn.Linear(vision_feature_dim + imu_plan_hidden_dim, vision_plan_dim)
        
        # --- 高頻度ブランチ（コントローラー）の定義 ---
        # 状態を更新しないため、RNNは不要。固定された状態とIMUサンプルを直接結合するMLPのみ。
        self.control_output_layer = nn.Sequential(
            nn.Linear(vision_plan_dim + 6, 64), # 固定状態 + IMU 6軸
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        self.dropout = nn.Dropout(dropout_p)
        
        # --- 推論用の内部状態 ---
        self.vision_plan_state = None
        self.device = torch.device("cpu")

    def to(self, device):
        super().to(device)
        self.device = device
        return self

    @torch.no_grad()
    def update_plan(self, image, imu_plan_sequence):
        self.eval()
        vision_features = self.vision_encoder(image).view(image.size(0), -1)
        vision_features = self.vision_fc(vision_features)
        _, (imu_plan_hidden, _) = self.imu_plan_encoder(imu_plan_sequence)
        imu_plan_features = imu_plan_hidden[-1]
        fused_features = torch.cat([vision_features, imu_plan_features], dim=1)
        
        # 計算した運転方針を内部状態として保存（固定）
        self.vision_plan_state = self.fusion_fc(fused_features)

    @torch.no_grad()
    def predict_control(self, imu_sample):
        self.eval()
        if self.vision_plan_state is None:
            # まだ方針がなければゼロ制御
            return 0.0, 0.0

        # 固定された運転方針と最新のIMUサンプルを結合
        # imu_sampleの形状は (1, 1, 6) なので、(1, 6) に変形
        control_input = torch.cat([self.vision_plan_state, imu_sample.squeeze(1)], dim=1)
        
        # MLPで制御量を計算
        control_output = self.control_output_layer(control_input)
        accelerator = torch.sigmoid(control_output[:, 0]).item()
        steering = torch.tanh(control_output[:, 1]).item()
        return accelerator, steering

    def forward(self, image, imu_sequence):
        # --- 学習時の動作 ---
        # 1. 低頻度ブランチで固定する運転方針を計算
        vision_features = self.vision_encoder(image).view(image.size(0), -1)
        vision_features = self.vision_fc(vision_features)
        _, (imu_plan_hidden, _) = self.imu_plan_encoder(imu_sequence)
        imu_plan_features = imu_plan_hidden[-1]
        fused_features = torch.cat([vision_features, imu_plan_features], dim=1)
        vision_plan_state = self.fusion_fc(fused_features)
        
        # 2. 高頻度ブランチの動作をシミュレート
        batch_size, seq_len, _ = imu_sequence.shape
        # vision_plan_stateをシーケンス長だけ複製し、各IMUサンプルと結合
        vision_plan_state_expanded = vision_plan_state.unsqueeze(1).repeat(1, seq_len, 1)
        control_inputs = torch.cat([vision_plan_state_expanded, imu_sequence], dim=2)
        
        # MLPはシーケンス全体を一度に処理できる
        control_outputs = self.control_output_layer(control_inputs)
        
        accelerator = torch.sigmoid(control_outputs[:, :, 0])
        steering = torch.tanh(control_outputs[:, :, 1])
        final_outputs = torch.stack([accelerator, steering], dim=-1)
        return final_outputs
    
# ==============================================================================
# モデルB: 更新あり（逐次更新）
# ==============================================================================
class ModelB_StateUpdating(nn.Module):
    def __init__(self, vision_feature_dim=512, imu_plan_hidden_dim=128, control_rnn_hidden_dim=64, dropout_p=0.2):
        super().__init__()
        self.control_rnn_hidden_dim = control_rnn_hidden_dim
        # --- 低頻度ブランチ（プランナー）の定義 ---
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.vision_encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.vision_fc = nn.Linear(resnet.fc.in_features, vision_feature_dim)
        self.imu_plan_encoder = nn.LSTM(input_size=6, hidden_size=imu_plan_hidden_dim, num_layers=2, batch_first=True, dropout=dropout_p)
        self.fusion_fc = nn.Linear(vision_feature_dim + imu_plan_hidden_dim, control_rnn_hidden_dim)
        
        # --- 高頻度ブランチ（コントローラー）の定義 ---
        self.control_rnn = nn.GRU(input_size=6, hidden_size=control_rnn_hidden_dim, num_layers=1, batch_first=True)
        self.control_output_layer = nn.Sequential(nn.Linear(control_rnn_hidden_dim, 32), nn.ReLU(), nn.Linear(32, 2))
        
        # --- 推論用の内部状態 ---
        self.control_hidden_state = None
        self.device = torch.device("cpu")

    def to(self, device):
        super().to(device)
        self.device = device
        return self

    @torch.no_grad()
    def update_plan(self, image, imu_plan_sequence):
        self.eval()
        vision_features = self.vision_encoder(image).view(image.size(0), -1)
        vision_features = self.vision_fc(vision_features)
        _, (imu_plan_hidden, _) = self.imu_plan_encoder(imu_plan_sequence)
        imu_plan_features = imu_plan_hidden[-1]
        fused_features = torch.cat([vision_features, imu_plan_features], dim=1)
        # 新しい運転方針（隠れ状態）を計算し、内部状態を上書き
        self.control_hidden_state = self.fusion_fc(fused_features).unsqueeze(0)

    @torch.no_grad()
    def predict_control(self, imu_sample):
        self.eval()
        if self.control_hidden_state is None:
            self.control_hidden_state = torch.zeros(1, 1, self.control_rnn_hidden_dim, device=self.device)
        
        # RNNで状態を更新しつつ、出力を得る
        rnn_output, next_hidden_state = self.control_rnn(imu_sample, self.control_hidden_state)
        # 内部状態を次の状態に更新
        self.control_hidden_state = next_hidden_state
        
        control_output = self.control_output_layer(rnn_output.squeeze(1))
        accelerator = torch.sigmoid(control_output[:, 0]).item()
        steering = torch.tanh(control_output[:, 1]).item()
        return accelerator, steering

    def forward(self, image, imu_sequence):
        # --- 学習時の動作 ---
        vision_features = self.vision_encoder(image).view(image.size(0), -1)
        vision_features = self.vision_fc(vision_features)
        _, (imu_plan_hidden, _) = self.imu_plan_encoder(imu_sequence)
        imu_plan_features = imu_plan_hidden[-1]
        fused_features = torch.cat([vision_features, imu_plan_features], dim=1)
        target_hidden_state = self.fusion_fc(fused_features).unsqueeze(0)
        
        control_rnn_outputs, _ = self.control_rnn(imu_sequence, target_hidden_state)
        control_outputs = self.control_output_layer(control_rnn_outputs)
        
        accelerator = torch.sigmoid(control_outputs[:, :, 0])
        steering = torch.tanh(control_outputs[:, :, 1])
        final_outputs = torch.stack([accelerator, steering], dim=-1)
        return final_outputs
    
# ==============================================================================
# モデルC: ハイブリッド（スキップ接続）
# ==============================================================================
class ModelC_HybridSkipConnection(nn.Module):
    def __init__(self, vision_feature_dim=512, imu_plan_hidden_dim=128, vision_plan_dim=64, control_rnn_hidden_dim=64, dropout_p=0.2):
        super().__init__()
        self.vision_plan_dim = vision_plan_dim
        self.control_rnn_hidden_dim = control_rnn_hidden_dim
        
        # --- 低頻度ブランチ（プランナー）の定義 ---
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.vision_encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.vision_fc = nn.Linear(resnet.fc.in_features, vision_feature_dim)
        self.imu_plan_encoder = nn.LSTM(input_size=6, hidden_size=imu_plan_hidden_dim, num_layers=2, batch_first=True, dropout=dropout_p)
        # 2つの出力を生成する層
        self.fusion_fc_vision = nn.Linear(vision_feature_dim + imu_plan_hidden_dim, vision_plan_dim)
        self.fusion_fc_control = nn.Linear(vision_feature_dim + imu_plan_hidden_dim, control_rnn_hidden_dim)

        # --- 高頻度ブランチ（コントローラー）の定義 ---
        self.control_rnn = nn.GRU(input_size=6, hidden_size=control_rnn_hidden_dim, num_layers=1, batch_first=True)
        # 制御出力層は、更新された状態と固定された視覚方針の両方を入力として受け取る
        self.control_output_layer = nn.Sequential(
            nn.Linear(control_rnn_hidden_dim + vision_plan_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        
        # --- 推論用の内部状態 ---
        self.vision_plan_state = None
        self.control_hidden_state = None
        self.device = torch.device("cpu")
        
    def to(self, device):
        super().to(device)
        self.device = device
        return self

    @torch.no_grad()
    def update_plan(self, image, imu_plan_sequence):
        self.eval()
        vision_features = self.vision_encoder(image).view(image.size(0), -1)
        vision_features = self.vision_fc(vision_features)
        _, (imu_plan_hidden, _) = self.imu_plan_encoder(imu_plan_sequence)
        imu_plan_features = imu_plan_hidden[-1]
        fused_features = torch.cat([vision_features, imu_plan_features], dim=1)
        
        # 2つの内部状態を更新
        self.vision_plan_state = self.fusion_fc_vision(fused_features)
        self.control_hidden_state = self.fusion_fc_control(fused_features).unsqueeze(0)

    @torch.no_grad()
    def predict_control(self, imu_sample):
        self.eval()
        if self.control_hidden_state is None or self.vision_plan_state is None:
            # 初期化されていなければゼロで初期化
            self.control_hidden_state = torch.zeros(1, 1, self.control_rnn_hidden_dim, device=self.device)
            self.vision_plan_state = torch.zeros(1, self.vision_plan_dim, device=self.device)

        # 1. IMUによる短期状態の更新
        _, next_hidden_state = self.control_rnn(imu_sample, self.control_hidden_state)
        self.control_hidden_state = next_hidden_state
        
        # 2. 制御量の計算
        # 更新された短期状態と、固定された視覚方針を結合（スキップ接続）
        combined_input = torch.cat([next_hidden_state.squeeze(0), self.vision_plan_state], dim=1)
        
        control_output = self.control_output_layer(combined_input)
        accelerator = torch.sigmoid(control_output[:, 0]).item()
        steering = torch.tanh(control_output[:, 1]).item()
        return accelerator, steering

    def forward(self, image, imu_sequence):
        # --- 学習時の動作 ---
        # 1. 低頻度ブランチで2つの初期状態を計算
        vision_features = self.vision_encoder(image).view(image.size(0), -1)
        vision_features = self.vision_fc(vision_features)
        _, (imu_plan_hidden, _) = self.imu_plan_encoder(imu_sequence)
        imu_plan_features = imu_plan_hidden[-1]
        fused_features = torch.cat([vision_features, imu_plan_features], dim=1)
        
        vision_plan_state = self.fusion_fc_vision(fused_features)
        initial_control_hidden = self.fusion_fc_control(fused_features).unsqueeze(0)
        
        # 2. 高頻度RNNで短期状態のシーケンスを計算
        control_rnn_outputs, _ = self.control_rnn(imu_sequence, initial_control_hidden)
        
        # 3. スキップ接続をシミュレート
        batch_size, seq_len, _ = imu_sequence.shape
        vision_plan_state_expanded = vision_plan_state.unsqueeze(1).repeat(1, seq_len, 1)
        combined_inputs = torch.cat([control_rnn_outputs, vision_plan_state_expanded], dim=2)
        
        control_outputs = self.control_output_layer(combined_inputs)
        accelerator = torch.sigmoid(control_outputs[:, :, 0])
        steering = torch.tanh(control_outputs[:, :, 1])
        final_outputs = torch.stack([accelerator, steering], dim=-1)
        return final_outputs