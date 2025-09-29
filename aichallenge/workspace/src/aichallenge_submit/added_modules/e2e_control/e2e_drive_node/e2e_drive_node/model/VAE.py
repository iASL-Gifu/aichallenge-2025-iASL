import torch
from torch import nn
from typing import List, Tuple

class ConvEncoder(nn.Module):
    """
    ご提示のVAEのエンコーダ部分を再利用可能なモジュールとしてクラス化。
    """
    def __init__(self,
                 in_channels: int,
                 hidden_dims: List[int],
                 img_size: Tuple[int, int]):
        super().__init__()

        modules = []
        current_channels = in_channels
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(current_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.ReLU())
            )
            current_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        
        # ダウンサンプリング後の特徴マップサイズを計算
        h, w = img_size
        num_downsample = len(hidden_dims)
        self.final_h = h // (2 ** num_downsample)
        self.final_w = w // (2 ** num_downsample)
        
        if self.final_h <= 0 or self.final_w <= 0:
            raise ValueError(f"入力サイズ({img_size})と層の数({num_downsample})が不適切です。")

        self.output_channels = hidden_dims[-1]
        self.flattened_size = self.output_channels * self.final_h * self.final_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

class ConvDecoder(nn.Module):
    """
    ご提示のVAEのデコーダ部分を再利用可能なモジュールとしてクラス化。
    """
    def __init__(self,
                 latent_dim: int,
                 hidden_dims: List[int],
                 encoder_output_shape: Tuple[int, int, int],
                 output_channels: int,
                 output_activation: nn.Module = nn.Sigmoid()):
        super().__init__()

        self.encoder_output_channels, self.final_h, self.final_w = encoder_output_shape
        self.flattened_size = self.encoder_output_channels * self.final_h * self.final_w

        self.decoder_input = nn.Linear(latent_dim, self.flattened_size)

        decoder_hidden_dims = list(reversed(hidden_dims))
        
        modules = []
        current_channels = decoder_hidden_dims[0]
        
        for i in range(len(decoder_hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(current_channels,
                                       decoder_hidden_dims[i + 1],
                                       kernel_size=3, stride=2,
                                       padding=1, output_padding=1),
                    nn.BatchNorm2d(decoder_hidden_dims[i + 1]),
                    nn.ReLU())
            )
            current_channels = decoder_hidden_dims[i + 1]
            
        # 最後の転置畳み込み層
        modules.append(
             nn.Sequential(
                nn.ConvTranspose2d(current_channels,
                                   hidden_dims[0], # 元のhidden_dims[0]
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   output_padding=1),
                nn.BatchNorm2d(hidden_dims[0]),
                nn.ReLU())
        )

        self.decoder = nn.Sequential(*modules)
        
        # 最終出力層
        self.final_layer = nn.Sequential(
            nn.Conv2d(hidden_dims[0], out_channels=output_channels,
                      kernel_size=3, padding=1),
        )
        if output_activation:
             self.final_layer.add_module("Activation", output_activation)


    def forward(self, z: torch.Tensor) -> torch.Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, self.encoder_output_channels, self.final_h, self.final_w)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result


class VAE(nn.Module):
    """
    ConvEncoderとConvDecoderを使用して再構築されたVAEモデル。
    """
    def __init__(self,
                 in_channels: int = 3,
                 latent_dim: int = 64,
                 hidden_dims: List[int] = None,
                 img_size: Tuple[int, int] = (272, 480)) -> None:
        super().__init__()

        self.latent_dim = latent_dim

        if hidden_dims is None:
            hidden_dims = [16, 32, 64, 128]

        # --- エンコーダ ---
        self.encoder = ConvEncoder(in_channels=in_channels,
                                   hidden_dims=hidden_dims,
                                   img_size=img_size)

        # --- 潜在変数への射影 ---
        self.fc_mu = nn.Linear(self.encoder.flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(self.encoder.flattened_size, latent_dim)

        # --- デコーダ ---
        self.decoder = ConvDecoder(
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            encoder_output_shape=(self.encoder.output_channels, self.encoder.final_h, self.encoder.final_w),
            output_channels=in_channels,
            output_activation=nn.Sigmoid()
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """エンコーダ: 入力xから潜在変数のパラメータmuとlogvarを計算"""
        feat = self.encoder(x)
        feat = torch.flatten(feat, start_dim=1)
        mu = self.fc_mu(feat)
        logvar = self.fc_logvar(feat)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """再パラメータ化トリック: muとlogvarから潜在変数zをサンプリング"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """デコーダ: 潜在変数zから画像を復元"""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """順伝播"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar
    
class SegVAE(nn.Module):
    """
    ConvEncoderとConvDecoderを使用して構築されたセグメンテーションマスク用のVAE。
    入力はone-hot化されたマスクテンソルを想定。
    出力は再構成マスクのロジット（CrossEntropyLoss用）。
    """
    def __init__(self,
                 mask_channels: int,
                 latent_dim: int = 64,
                 hidden_dims: List[int] = None,
                 img_size: Tuple[int, int] = (272, 480)) -> None:
        super().__init__()

        self.latent_dim = latent_dim

        if hidden_dims is None:
            hidden_dims = [16, 32, 64, 128]

        # --- エンコーダ (マスク用) ---
        # 入力チャンネル数はマスクのクラス数になります
        self.encoder = ConvEncoder(in_channels=mask_channels,
                                   hidden_dims=hidden_dims,
                                   img_size=img_size)

        # --- 潜在変数への射影 ---
        self.fc_mu = nn.Linear(self.encoder.flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(self.encoder.flattened_size, latent_dim)

        # --- デコーダ (マスク用) ---
        self.decoder = ConvDecoder(
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            encoder_output_shape=(self.encoder.output_channels, self.encoder.final_h, self.encoder.final_w),
            output_channels=mask_channels,
            output_activation=None
        )

    def encode(self, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """エンコーダ: 入力マスクwから潜在変数のパラメータmuとlogvarを計算"""
        feat = self.encoder(w)
        feat = torch.flatten(feat, start_dim=1)
        mu = self.fc_mu(feat)
        logvar = self.fc_logvar(feat)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """再パラメータ化トリック: muとlogvarから潜在変数zをサンプリング"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """デコーダ: 潜在変数zからマスクのロジットを復元"""
        return self.decoder(z)

    def forward(self, w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """順伝播"""
        mu, logvar = self.encode(w)
        z = self.reparameterize(mu, logvar)
        reconstruction_logits = self.decode(z)
        return reconstruction_logits, mu, logvar

class JMVAE(nn.Module):
    def __init__(self,
                 latent_dim: int = 128,
                 img_channels: int = 3,
                 mask_channels: int = 4, # セマンティックマスクのクラス数
                 img_size: Tuple[int, int] = (272, 480),
                 hidden_dims: List[int] = None):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [16, 32, 64, 128]

        # --- 画像 (x) 用のエンコーダ・デコーダ ---
        self.encoder_x = ConvEncoder(in_channels=img_channels, hidden_dims=hidden_dims, img_size=img_size)
        self.decoder_x = ConvDecoder(latent_dim=latent_dim, hidden_dims=hidden_dims,
                                     encoder_output_shape=(self.encoder_x.output_channels, self.encoder_x.final_h, self.encoder_x.final_w),
                                     output_channels=img_channels,
                                     output_activation=nn.Sigmoid())

        # --- マスク (w) 用のエンコーダ・デコーダ ---
        self.encoder_w = ConvEncoder(in_channels=mask_channels, hidden_dims=hidden_dims, img_size=img_size)
        self.decoder_w = ConvDecoder(latent_dim=latent_dim, hidden_dims=hidden_dims,
                                     encoder_output_shape=(self.encoder_w.output_channels, self.encoder_w.final_h, self.encoder_w.final_w),
                                     output_channels=mask_channels,
                                     output_activation=None) # CrossEntropyLossを使うため活性化関数なし

        # --- 潜在変数を生成するヘッド部分 ---
        joint_feature_dim = self.encoder_x.flattened_size + self.encoder_w.flattened_size

        # 共同エンコーダ q(z|x,w)
        self.fc_mu_joint = nn.Linear(joint_feature_dim, latent_dim)
        self.fc_logvar_joint = nn.Linear(joint_feature_dim, latent_dim)

        # 単独エンコーダ q(z|x)
        self.fc_mu_x = nn.Linear(self.encoder_x.flattened_size, latent_dim)
        self.fc_logvar_x = nn.Linear(self.encoder_x.flattened_size, latent_dim)

        # 単独エンコーダ q(z|w)
        self.fc_mu_w = nn.Linear(self.encoder_w.flattened_size, latent_dim)
        self.fc_logvar_w = nn.Linear(self.encoder_w.flattened_size, latent_dim)


    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> Tuple:
        # 1. 各モダリティの特徴を抽出
        feat_x = self.encoder_x(x)
        feat_w = self.encoder_w(w)
        
        flat_feat_x = torch.flatten(feat_x, start_dim=1)
        flat_feat_w = torch.flatten(feat_w, start_dim=1)

        # 2. 共同エンコーダ q(z|x,w)
        feat_joint = torch.cat([flat_feat_x, flat_feat_w], dim=1)
        mu_joint = self.fc_mu_joint(feat_joint)
        logvar_joint = self.fc_logvar_joint(feat_joint)
        
        # 3. 単独エンコーダ q(z|x) と q(z|w)
        mu_x = self.fc_mu_x(flat_feat_x)
        logvar_x = self.fc_logvar_x(flat_feat_x)
        mu_w = self.fc_mu_w(flat_feat_w)
        logvar_w = self.fc_logvar_w(flat_feat_w)
        
        # 4. 潜在変数 z をサンプリング (共同表現から)
        z = self.reparameterize(mu_joint, logvar_joint)
        
        # 5. 各モダリティをデコード
        recon_x = self.decoder_x(z)
        recon_w_logits = self.decoder_w(z) # Softmax前のlogits
        
        return (recon_x, recon_w_logits, 
                mu_joint, logvar_joint,
                mu_x, logvar_x,
                mu_w, logvar_w)