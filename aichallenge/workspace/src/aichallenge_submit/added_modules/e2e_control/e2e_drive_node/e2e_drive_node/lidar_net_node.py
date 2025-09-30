import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np
import time

# --- PyTorch関連のインポート ---
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- ROS2メッセージの型をインポート ---
from sensor_msgs.msg import LaserScan
from autoware_auto_control_msgs.msg import AckermannControlCommand
from std_msgs.msg import Empty

# --- PyTorchモデルの定義 ---

class TinyLidarNet(nn.Module):
    """ ★★★ PyTorch版 従来モデル ★★★ """
    def __init__(self, input_dim=1080, output_dim=2):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 24, kernel_size=10, stride=4)
        self.conv2 = nn.Conv1d(24, 36, kernel_size=8, stride=4)
        self.conv3 = nn.Conv1d(36, 48, kernel_size=4, stride=2)
        self.conv4 = nn.Conv1d(48, 64, kernel_size=3)
        self.conv5 = nn.Conv1d(64, 64, kernel_size=3)
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_dim)
            x = self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(dummy_input)))))
            flatten_dim = x.view(1, -1).shape[1]
        self.fc1 = nn.Linear(flatten_dim, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, output_dim)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        return x

class TinyLidarNetSmall(nn.Module):
    """ ★★★ PyTorch版 軽量モデル ★★★ """
    def __init__(self, input_dim=270, output_dim=2):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=10, stride=4)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=8, stride=4)
        self.conv3 = nn.Conv1d(32, 48, kernel_size=4, stride=2)
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_dim)
            x = self.conv3(self.conv2(self.conv1(dummy_input)))
            flatten_dim = x.view(1, -1).shape[1]
        self.fc1 = nn.Linear(flatten_dim, 64)
        self.fc2 = nn.Linear(64, 10)
        self.fc3 = nn.Linear(10, output_dim)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x


class E2ENode(Node):
    def __init__(self):
        super().__init__('e2e_node_py')

        # === パラメータ宣言 ===
        self.declare_parameter('log_interval_sec', 5.0)
        self.declare_parameter('model.input_dim', 1080)
        self.declare_parameter('model.output_dim', 2) 
        self.declare_parameter('model.architecture', 'large') 
        self.declare_parameter('model.weight_path', '') 
        self.declare_parameter('acceleration', 0.1) 
        
        # === パラメータ取得 ===
        self.log_interval = self.get_parameter('log_interval_sec').get_parameter_value().double_value
        self.input_dim = self.get_parameter('model.input_dim').get_parameter_value().integer_value
        self.output_dim = self.get_parameter('model.output_dim').get_parameter_value().integer_value
        self.model_architecture = self.get_parameter('model.architecture').get_parameter_value().string_value 
        model_weight_path = self.get_parameter('model.weight_path').get_parameter_value().string_value
        self.acceleration = self.get_parameter('acceleration').get_parameter_value().double_value

        self.get_logger().info(f"Model architecture is set to: '{self.model_architecture}'") 

        # === デバイス設定 ===
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f"Using device: {self.device}")

        # === PyTorchモデルのロード ===
        ModelClass = None
        if self.model_architecture == 'small':
            ModelClass = TinyLidarNetSmall
        else: 
            ModelClass = TinyLidarNet

        if not model_weight_path:
            self.get_logger().fatal("Model weight path is not set.")
            rclpy.shutdown(); return
        try:
            # モデルクラスをインスタンス化
            self.model = ModelClass(input_dim=self.input_dim, output_dim=self.output_dim)
            # 学習済み重みをロード
            self.model.load_state_dict(torch.load(model_weight_path, map_location=self.device))
            # モデルを評価モードに設定
            self.model.eval()
            # モデルを適切なデバイスに転送
            self.model.to(self.device)
            self.get_logger().info(f"Model ({self.model_architecture}) loaded from {model_weight_path}")
        except Exception as e:
            self.get_logger().fatal(f"Model file not found or structure mismatch: {e}")
            rclpy.shutdown(); return

        # === 変数の初期化 ===
        self.inference_times = []
        self.last_log_time = self.get_clock().now()

        # === 通信設定 ===
        sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(LaserScan, "/scan", self.scan_callback, sensor_qos)
        self.create_subscription(Empty, "/aichallenge/awsim/reset", self.reset_callback, 10)
        self.control_pub = self.create_publisher(AckermannControlCommand, "/awsim/control_cmd", 1)
        
        self.get_logger().info("E2E Python Node (PyTorch version) has been initialized successfully.")
        
    def scan_callback(self, msg):
        start_time = time.monotonic()
        
        # === データ前処理 (NumPy) ===
        ranges = np.array(msg.ranges, dtype=np.float32)
        ranges[np.isinf(ranges)] = msg.range_max
        ranges[np.isnan(ranges)] = 0.0
        
        mode_str = f"AI-{self.model_architecture}"
        processed_ranges = self.process_ranges_for_model(ranges)

        # === PyTorchでの推論 ===
        with torch.no_grad(): # 勾配計算を無効化して高速化
            # NumPy配列をPyTorchテンソルに変換し、デバイスに送る
            scan_tensor = torch.from_numpy(processed_ranges).unsqueeze(0).to(self.device)
            
            # モデルでフォワードパスを実行
            outputs = self.model(scan_tensor)
            
            # 結果をCPUに戻し、Pythonの数値に変換
            steering = outputs[0, 1].item()
        
        steering = np.clip(steering, -1.0, 1.0)
        
        # 推論時間計測
        end_time = time.monotonic()
        duration_ms = (end_time - start_time) * 1000.0
        self.inference_times.append(duration_ms)
        
        # 制御コマンド生成
        cmd = AckermannControlCommand()
        cmd.stamp = self.get_clock().now().to_msg()
        cmd.longitudinal.acceleration = self.acceleration
        cmd.lateral.steering_tire_angle = float(steering)
        self.control_pub.publish(cmd)
        
        # ログ出力
        self.get_logger().info(f"[{mode_str}] Published: Steer={steering:.3f} (Accel CMD: {cmd.longitudinal.acceleration:.4f})", throttle_duration_sec=1.0)
        
        # 定期的な統計情報ログ
        current_time = self.get_clock().now()
        if (current_time - self.last_log_time).nanoseconds / 1e9 > self.log_interval:
            if self.inference_times:
                avg_time = np.mean(self.inference_times)
                max_time = np.max(self.inference_times)
                avg_hz = 1000.0 / avg_time if avg_time > 0 else float('inf')
                self.get_logger().info(f"--- Inference Stats (last {self.log_interval:.1f}s) ---")
                self.get_logger().info(f"  Avg Time: {avg_time:.2f} ms | Avg Hz: {avg_hz:.2f} Hz | Max Time: {max_time:.2f} ms")
                self.get_logger().info(f"  Samples: {len(self.inference_times)}")
                self.inference_times.clear()
            self.last_log_time = current_time

    def process_ranges_for_model(self, ranges):
        # (NumPy版から変更なし)
        current_len = len(ranges)
        if current_len > self.input_dim:
            start_index = (current_len - self.input_dim) // 2
            processed = ranges[start_index : start_index + self.input_dim]
        elif current_len < self.input_dim:
            pad_width = self.input_dim - current_len
            processed = np.pad(ranges, (0, pad_width), 'constant', constant_values=0)
        else:
            processed = ranges
        processed /= 30.0
        return processed

    def reset_callback(self, msg):
        self.get_logger().warn("<<<<< Received reset signal. Clearing inference stats. >>>>>")
        self.inference_times.clear()

def main(args=None):
    rclpy.init(args=args)
    node = E2ENode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()