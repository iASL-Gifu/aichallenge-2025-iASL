import torch
import torch.nn as nn
import math
from typing import Optional

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, 1, embed_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1)].transpose(0, 1)
        return self.dropout(x)

class SensorFusionTransformer(nn.Module):
    def __init__(self,
                 sequence_length: int,
                 use_image: bool = True,
                 vae: Optional[nn.Module] = None,
                 use_imu: bool = True,
                 use_gnss: bool = True,
                 use_vehicle_status: bool = True,
                 output_dim: int = 2,
                 embed_dim: int = 128,
                 num_heads: int = 4,
                 num_layers: int = 3,
                 dropout_prob: float = 0.1
                 ) -> None:
        super().__init__()
        
        if use_image and vae is None:
            raise ValueError("use_image=True の場合、VAEモデルを渡す必要があります。")

        self.use_image = use_image
        self.use_imu = use_imu
        self.use_gnss = use_gnss
        self.use_vehicle_status = use_vehicle_status
        
        img_dim = 0
        if self.use_image:
            self.vae = vae
            self.vae.eval()
            for param in self.vae.parameters():
                param.requires_grad = False
            img_dim = self.vae.latent_dim

        # --- 各特徴量の次元数を動的に計算 ---
        imu_dim = 10 if use_imu else 0
        gnss_dim = 2 if use_gnss else 0
        vehicle_status_dim = 4 if use_vehicle_status else 0
        
        total_input_dim = img_dim + imu_dim + gnss_dim + vehicle_status_dim
        if total_input_dim == 0:
            raise ValueError("少なくとも1つのセンサーまたは画像を使用してください。")

        self.input_projection = nn.Linear(total_input_dim, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout_prob, max_len=sequence_length)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout_prob, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, 64), nn.ReLU(),
            nn.Dropout(dropout_prob), nn.Linear(64, output_dim)
        )

    def forward(self,
                image_seq: Optional[torch.Tensor] = None,
                imu_seq: Optional[torch.Tensor] = None,
                positions_seq: Optional[torch.Tensor] = None,
                velocity_seq: Optional[torch.Tensor] = None,
                steering_seq: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        
        features_to_concatenate = []

        if self.use_image:
            if image_seq is None:
                raise ValueError("use_image=True の場合、forwardメソッドに image_seq を渡す必要があります。")
            
            with torch.no_grad():
                b, s, c, h, w = image_seq.shape
                image_seq_reshaped = image_seq.view(b * s, c, h, w)
                mu, _ = self.vae.encode(image_seq_reshaped)
                image_features_seq = mu.view(b, s, -1)
            
            features_to_concatenate.append(image_features_seq)

        if self.use_imu and imu_seq is not None:
            features_to_concatenate.append(imu_seq)
        if self.use_gnss and positions_seq is not None:
            features_to_concatenate.append(positions_seq)
        if self.use_vehicle_status and velocity_seq is not None and steering_seq is not None:
            vehicle_data = torch.cat([velocity_seq, steering_seq], dim=2)
            features_to_concatenate.append(vehicle_data)

        if not features_to_concatenate:
            raise ValueError("入力データが1つも提供されませんでした。")
            
        combined_features = torch.cat(features_to_concatenate, dim=2)
        
        x = self.input_projection(combined_features)
        x = self.pos_encoder(x)
        transformer_output = self.transformer_encoder(x)
        last_step_output = transformer_output[:, -1, :]
        output = self.regressor(last_step_output)
        
        return output