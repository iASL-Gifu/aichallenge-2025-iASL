import torch
import torch.nn as nn
from copy import deepcopy

class ModelEMA:
    """
    モデルの重みの指数移動平均を管理するクラス
    """
    def __init__(self, model, decay=0.999):
        # 元のモデルと全く同じ構造のモデルをディープコピーして保持
        self.ema_model = deepcopy(model)
        self.ema_model.eval()  # EMAモデルは評価モードにする
        self.decay = decay
        
        # EMAモデルのパラメータは勾配計算を不要にする
        for param in self.ema_model.parameters():
            param.requires_grad_(False)

    def update(self, model):
        """
        現在のモデルの重みを使って、EMAモデルの重みを更新する
        このメソッドは optimizer.step() の直後に呼ばれる
        """
        with torch.no_grad():
            # EMAモデルの全パラメータを取得
            ema_params = self.ema_model.state_dict()
            
            # 現在のモデルの全パラメータをループ
            for name, param in model.state_dict().items():
                if param.requires_grad:
                    # EMAの更新式を適用
                    ema_params[name].data.mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    # 推論時にEMAモデルを直接使えるように、__call__を定義することも可能
    def __call__(self, *args, **kwargs):
        return self.ema_model(*args, **kwargs)