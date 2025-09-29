import torch
import numpy as np
import argparse
from src.model import TinyLidarNet, TinyLidarNetSmall 

def convert_weights_to_npz(model_class, pth_path, npz_path, input_dim):
    """
    PyTorchの .pth ファイルを NumPy の .npz 形式に変換します。
    
    Args:
        model_class (torch.nn.Module): モデルのクラス (例: TinyLidarNet)
        pth_path (str): 入力となる .pth ファイルのパス
        npz_path (str): 出力先の .npz ファイルのパス
        input_dim (int): モデルの入力次元
    """
    print(f"Converting '{pth_path}' to '{npz_path}'...")
    
    try:
        # モデルのインスタンス化とCPUへのマッピング
        device = torch.device('cpu')
        model = model_class(input_dim=input_dim, output_dim=2)
        
        # .pth ファイルから重みをロード
        model.load_state_dict(torch.load(pth_path, map_location=device))
        model.eval()

        # モデルのパラメータをNumPy配列に変換
        weights_numpy = {name: param.detach().cpu().numpy() for name, param in model.state_dict().items()}

        # .npz形式で保存
        np.savez(npz_path, **weights_numpy)
        
        print(f"✅ Conversion successful. Saved to '{npz_path}'")

    except FileNotFoundError:
        print(f"❌ Error: Input file not found at '{pth_path}'")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")

def main():
    """
    コマンドライン引数を処理し、モデル変換を実行します。
    """
    parser = argparse.ArgumentParser(
        description="Convert PyTorch .pth model weights to NumPy .npz format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--model-type',
        type=str,
        required=True,
        choices=['small', 'large'],
        help="Specify the model architecture type."
    )
    parser.add_argument(
        '--pth-path',
        type=str,
        required=True,
        help="Path to the input .pth weight file."
    )
    parser.add_argument(
        '--npz-path',
        type=str,
        required=True,
        help="Path for the output .npz weight file."
    )

    args = parser.parse_args()

    # モデルタイプに応じてクラスと入力次元を決定
    if args.model_type == 'small':
        model_class = TinyLidarNetSmall
        input_dim = 270
    else:  # 'large'
        model_class = TinyLidarNet
        input_dim = 1080
    
    print(f"Selected model: {model_class.__name__} (input_dim set to {input_dim})")

    # 変換関数を呼び出し
    convert_weights_to_npz(
        model_class=model_class,
        pth_path=args.pth_path,
        npz_path=args.npz_path,
        input_dim=input_dim
    )

if __name__ == '__main__':
    main()