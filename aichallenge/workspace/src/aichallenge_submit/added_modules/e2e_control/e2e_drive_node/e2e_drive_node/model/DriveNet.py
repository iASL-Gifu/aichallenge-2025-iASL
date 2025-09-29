import torch
from torch import nn
from typing import Optional

class DriveNet(nn.Module):
    """
    画像特徴と複数の追加センサデータを結合して推論を行うモデル。
    画像を含む各センサの利用有無をTrue/Falseで選択できる。
    """
    def __init__(self,
                 vae: Optional[nn.Module] = None,
                 use_image: bool = True,
                 use_imu: bool = False,
                 use_gnss: bool = False,
                 use_vehicle_status: bool = False,
                 output_dim: int = 2) -> None:
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

        self.regressor = nn.Sequential(
            nn.Linear(total_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self,
                image: Optional[torch.Tensor] = None,
                imu_raw: Optional[torch.Tensor] = None,
                positions: Optional[torch.Tensor] = None,
                velocity_status: Optional[torch.Tensor] = None,
                steering_status: Optional[torch.Tensor] = None) -> torch.Tensor:
        features_to_concatenate = []
        
        if self.use_image:
            if image is None:
                raise ValueError("Image tensor must be provided when use_image is True.")
            mu, _ = self.vae.encode(image)
            features_to_concatenate.append(mu)

        if self.use_imu:
            if imu_raw is None:
                raise ValueError("IMU tensor must be provided when use_imu is True.")
            features_to_concatenate.append(imu_raw)
        
        if self.use_gnss:
            if positions is None:
                raise ValueError("Positions tensor must be provided when use_gnss is True.")
            features_to_concatenate.append(positions)
        
        if self.use_vehicle_status:
            if velocity_status is None or steering_status is None:
                raise ValueError("Vehicle status tensors must be provided when use_vehicle_status is True.")
            vehicle_data = torch.cat([velocity_status, steering_status], dim=1)
            features_to_concatenate.append(vehicle_data)
        
        if not features_to_concatenate:
            raise RuntimeError("No features to concatenate. Check model configuration and data input.")

        combined_features = torch.cat(features_to_concatenate, dim=1)
        output = self.regressor(combined_features)
        return output

class SegDriveNet(nn.Module):
    """
    セグメンテーションマスク特徴と複数の追加センサデータを結合して推論を行うモデル。
    マスクを含む各センサの利用有無を選択可能。
    """
    def __init__(self,
                 seg_vae: Optional[nn.Module] = None,
                 use_mask: bool = True,
                 use_imu: bool = False,
                 use_gnss: bool = False,
                 use_vehicle_status: bool = False,
                 output_dim: int = 2) -> None:
        """
        Args:
            seg_vae (Optional[nn.Module]): 事前学習済みのSegVAE. use_mask=Trueの場合に必須.
            use_mask (bool): セグメンテーションマスクを使用するか.
            ... (other args) ...
        """
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

        self.regressor = nn.Sequential(
            nn.Linear(total_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self,
                mask: Optional[torch.Tensor] = None,
                imu_raw: Optional[torch.Tensor] = None,
                positions: Optional[torch.Tensor] = None,
                velocity_status: Optional[torch.Tensor] = None,
                steering_status: Optional[torch.Tensor] = None) -> torch.Tensor:
        features_to_concatenate = []
        
        if self.use_mask:
            if mask is None:
                raise ValueError("Mask tensor must be provided when use_mask is True.")
            mu, _ = self.seg_vae.encode(mask)
            features_to_concatenate.append(mu)

        if self.use_imu:
            if imu_raw is None:
                raise ValueError("IMU tensor must be provided when use_imu is True.")
            features_to_concatenate.append(imu_raw)
        
        if self.use_gnss:
            if positions is None:
                raise ValueError("Positions tensor must be provided when use_gnss is True.")
            features_to_concatenate.append(positions)
        
        if self.use_vehicle_status:
            if velocity_status is None or steering_status is None:
                raise ValueError("Vehicle status tensors must be provided when use_vehicle_status is True.")
            vehicle_data = torch.cat([velocity_status, steering_status], dim=1)
            features_to_concatenate.append(vehicle_data)

        if not features_to_concatenate:
            raise RuntimeError("No features to concatenate. Check model configuration and data input.")

        combined_features = torch.cat(features_to_concatenate, dim=1)
        output = self.regressor(combined_features)
        return output

class DriveNetJM(nn.Module):
    """
    事前学習済みJMVAEと複数のセンサデータを結合して推論を行うモデル。
    JMVAEは入力(画像/マスク)に応じてエンコーダを自動で切り替える。
    各追加センサの利用有無をTrue/Falseで選択可能。
    """
    def __init__(self,
                 jmvae: nn.Module,
                 use_imu: bool = False,
                 use_gnss: bool = False,
                 use_vehicle_status: bool = False,
                 output_dim: int = 2) -> None:
        """
        Args:
            jmvae (JMVAE): 事前学習済みのJMVAEモデル
            use_imu (bool): IMUデータを使用するか
            use_gnss (bool): GNSSデータを使用するか
            use_vehicle_status (bool): 車両ステータスデータを使用するか
            output_dim (int): 出力次元数 (throttle, steer)
        """
        super().__init__()

        # JMVAEをセットし、パラメータを凍結
        self.jmvae = jmvae
        for param in self.jmvae.parameters():
            param.requires_grad = False

        # 画像/マスク特徴の次元をJMVAEから取得
        self.img_feature_dim = self.jmvae.latent_dim

        # 使用するセンサの次元を動的に加算
        total_input_dim = self.img_feature_dim
        self.use_imu = use_imu
        self.use_gnss = use_gnss
        self.use_vehicle_status = use_vehicle_status

        if self.use_imu:
            self.imu_dim = 10
            total_input_dim += self.imu_dim
        else:
            self.imu_dim = 0

        if self.use_gnss:
            self.gnss_dim = 2
            total_input_dim += self.gnss_dim
        else:
            self.gnss_dim = 0

        if self.use_vehicle_status:
            self.vehicle_status_dim = 4 # 速度と舵角情報など
            total_input_dim += self.vehicle_status_dim
        else:
            self.vehicle_status_dim = 0

        # 全ての特徴量を結合した後の回帰層
        self.regressor = nn.Sequential(
            nn.Linear(total_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self,
                image: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None,
                imu_raw: Optional[torch.Tensor] = None,
                positions: Optional[torch.Tensor] = None,
                velocity_status: Optional[torch.Tensor] = None,
                steering_status: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        順伝播: 入力に応じて特徴量を結合
        """
        # 1. JMVAEから画像/マスク特徴を抽出
        mu, _ = self.jmvae.encode(x=image, w=mask)

        # 2. 抽出した特徴量をベースとして、利用可能なセンサデータを結合
        features_to_concatenate = [mu]

        if self.use_imu and imu_raw is not None:
            features_to_concatenate.append(imu_raw)

        if self.use_gnss and positions is not None:
            features_to_concatenate.append(positions)

        if self.use_vehicle_status and velocity_status is not None and steering_status is not None:
            vehicle_data = torch.cat([velocity_status, steering_status], dim=1)
            features_to_concatenate.append(vehicle_data)

        # 3. すべての特徴を結合
        combined_features = torch.cat(features_to_concatenate, dim=1)

        # 4. 推論を実行
        output = self.regressor(combined_features)
        return output