import torch
from torch import nn
from typing import Tuple, Optional

class DriveNetRNN(nn.Module):
    """
    RNNの入出力次元を統一したモデル。
    画像を含む各センサの利用有無を選択可能。
    """
    def __init__(self,
                 vae: Optional[nn.Module] = None,
                 use_image: bool = True,
                 use_imu: bool = False,
                 use_gnss: bool = False,
                 use_vehicle_status: bool = False,
                 output_dim: int = 2,
                 rnn_num_layers: int = 1
                 ) -> None:
        super().__init__()
        self.use_image = use_image
        self.use_imu = use_imu
        self.use_gnss = use_gnss
        self.use_vehicle_status = use_vehicle_status
        self.vae = vae
        
        total_input_dim = 0
        
        if self.use_image:
            if self.vae is None:
                raise ValueError("VAE model must be provided when use_image is True.")
            self.img_feature_dim = self.vae.latent_dim
            total_input_dim += self.img_feature_dim

        if self.use_imu:
            self.imu_dim = 10
            total_input_dim += self.imu_dim
            
        if self.use_gnss:
            self.gnss_dim = 2
            total_input_dim += self.gnss_dim
            
        if self.use_vehicle_status:
            self.vehicle_status_dim = 4
            total_input_dim += self.vehicle_status_dim

        if total_input_dim == 0:
            raise ValueError("At least one input source must be enabled.")

        self.rnn = nn.LSTM(
            input_size=total_input_dim,
            hidden_size=total_input_dim,  
            num_layers=rnn_num_layers,
            batch_first=True
        )

        self.regressor = nn.Sequential(
            nn.Linear(total_input_dim, 64), 
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self,
                image_seq: Optional[torch.Tensor] = None,
                imu_raw_seq: Optional[torch.Tensor] = None,
                positions_seq: Optional[torch.Tensor] = None,
                velocity_status_seq: Optional[torch.Tensor] = None,
                steering_status_seq: Optional[torch.Tensor] = None,
                state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        
        features_to_concatenate = []
        
        if self.use_image:
            if image_seq is None:
                raise ValueError("Image sequence tensor must be provided when use_image is True.")
            batch_size, seq_len = image_seq.shape[:2]
            image_reshaped = image_seq.view(batch_size * seq_len, *image_seq.shape[2:])
            mu, _ = self.vae.encode(image_reshaped)
            img_features = mu.view(batch_size, seq_len, -1)
            features_to_concatenate.append(img_features)

        if self.use_imu and imu_raw_seq is not None:
            features_to_concatenate.append(imu_raw_seq)
        if self.use_gnss and positions_seq is not None:
            # GNSSデータが(x,y,z)などで来ても2次元にスライス
            features_to_concatenate.append(positions_seq[:, :, :self.gnss_dim])
        if self.use_vehicle_status and velocity_status_seq is not None and steering_status_seq is not None:
            vehicle_data = torch.cat([velocity_status_seq, steering_status_seq], dim=2)
            features_to_concatenate.append(vehicle_data)
        
        if not features_to_concatenate:
            raise RuntimeError("No features to concatenate. Check model configuration and data input.")

        combined_features = torch.cat(features_to_concatenate, dim=2)
        
        batch_size, seq_len = combined_features.shape[:2]
        device = combined_features.device

        if state is None:
            h_0 = torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size).to(device)
            c_0 = torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size).to(device)
            state = (h_0, c_0)

        rnn_out, new_state = self.rnn(combined_features, state)

        rnn_out_reshaped = rnn_out.reshape(batch_size * seq_len, self.rnn.hidden_size)
        output_reshaped = self.regressor(rnn_out_reshaped)
        
        output = output_reshaped.reshape(batch_size, seq_len, -1)
        return output, new_state
    

class SegDriveNetRNN(nn.Module):
    """
    SegVAEエンコーダとRNN (LSTM) を組み合わせた時系列モデル。
    マスクを含む各センサの利用有無を選択可能。
    """
    def __init__(self,
                 seg_vae: Optional[nn.Module] = None,
                 use_mask: bool = True,
                 use_imu: bool = False,
                 use_gnss: bool = False,
                 use_vehicle_status: bool = False,
                 output_dim: int = 2,
                 rnn_num_layers: int = 1
                 ) -> None:
        super().__init__()
        self.use_mask = use_mask
        self.use_imu = use_imu
        self.use_gnss = use_gnss
        self.use_vehicle_status = use_vehicle_status
        self.seg_vae = seg_vae
        
        total_input_dim = 0
        
        if self.use_mask:
            if self.seg_vae is None:
                raise ValueError("SegVAE model must be provided when use_mask is True.")
            self.mask_feature_dim = self.seg_vae.latent_dim
            total_input_dim += self.mask_feature_dim
        
        if self.use_imu:
            self.imu_dim = 10
            total_input_dim += self.imu_dim
            
        if self.use_gnss:
            self.gnss_dim = 2
            total_input_dim += self.gnss_dim
            
        if self.use_vehicle_status:
            self.vehicle_status_dim = 4
            total_input_dim += self.vehicle_status_dim

        if total_input_dim == 0:
            raise ValueError("At least one input source must be enabled.")

        self.rnn = nn.LSTM(
            input_size=total_input_dim,
            hidden_size=total_input_dim,  
            num_layers=rnn_num_layers,
            batch_first=True
        )

        self.regressor = nn.Sequential(
            nn.Linear(total_input_dim, 64), 
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self,
                mask_seq: Optional[torch.Tensor] = None,
                imu_raw_seq: Optional[torch.Tensor] = None,
                positions_seq: Optional[torch.Tensor] = None,
                velocity_status_seq: Optional[torch.Tensor] = None,
                steering_status_seq: Optional[torch.Tensor] = None,
                state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        
        features_to_concatenate = []

        if self.use_mask:
            if mask_seq is None:
                raise ValueError("Mask sequence tensor must be provided when use_mask is True.")
            batch_size, seq_len = mask_seq.shape[:2]
            mask_reshaped = mask_seq.view(batch_size * seq_len, *mask_seq.shape[2:])
            mu, _ = self.seg_vae.encode(mask_reshaped)
            mask_features = mu.view(batch_size, seq_len, -1)
            features_to_concatenate.append(mask_features)

        if self.use_imu and imu_raw_seq is not None:
            features_to_concatenate.append(imu_raw_seq)
        if self.use_gnss and positions_seq is not None:
            features_to_concatenate.append(positions_seq)
        if self.use_vehicle_status and velocity_status_seq is not None and steering_status_seq is not None:
            vehicle_data = torch.cat([velocity_status_seq, steering_status_seq], dim=2)
            features_to_concatenate.append(vehicle_data)
        
        if not features_to_concatenate:
            raise RuntimeError("No features to concatenate. Check model configuration and data input.")
        
        combined_features = torch.cat(features_to_concatenate, dim=2)
        
        batch_size, seq_len = combined_features.shape[:2]
        device = combined_features.device
        
        if state is None:
            h_0 = torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size).to(device)
            c_0 = torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size).to(device)
            state = (h_0, c_0)

        rnn_out, new_state = self.rnn(combined_features, state)

        rnn_out_reshaped = rnn_out.reshape(batch_size * seq_len, self.rnn.hidden_size)
        output_reshaped = self.regressor(rnn_out_reshaped)
        
        output = output_reshaped.reshape(batch_size, seq_len, -1)
        
        return output, new_state