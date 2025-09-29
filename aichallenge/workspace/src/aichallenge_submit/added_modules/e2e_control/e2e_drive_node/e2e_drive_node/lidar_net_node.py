# import rclpy
# from rclpy.node import Node
# from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
# import torch
# import numpy as np
# import math

# # --- ROS2メッセージの型をインポート ---
# from sensor_msgs.msg import LaserScan
# from autoware_auto_control_msgs.msg import AckermannControlCommand
# from geometry_msgs.msg import PoseWithCovarianceStamped
# from visualization_msgs.msg import Marker
# from std_msgs.msg import Empty # ▼▼▼【追加】リセット信号用のEmptyメッセージをインポート ▼▼▼

# # --- モデルのインポート ---
# from .model.TinyLidarNet import TinyLidarNet

# class TinyLidarNetNode(Node):
#     def __init__(self):
#         super().__init__('tiny_lidar_net_node')

#         # --- パラメータの宣言 ---
#         self.declare_parameter('timer_hz', 50.0)
#         self.declare_parameter('model.input_dim', 1080)
#         self.declare_parameter('model.output_dim', 2)
#         self.declare_parameter('model.pit_model_weight', 'pit_model.pth')
#         self.declare_parameter('model.outside_model_weight', 'outside_model.pth')
#         self.declare_parameter('pit_exit.x', 0.0)
#         self.declare_parameter('pit_exit.y', 0.0)
#         self.declare_parameter('pit_exit.radius', 10.0)
#         self.declare_parameter('acceleration.pit', 1.0)
#         self.declare_parameter('acceleration.outside', 0.0001)

#         # --- パラメータの取得 ---
#         timer_hz = self.get_parameter('timer_hz').get_parameter_value().double_value
#         self.input_dim = self.get_parameter('model.input_dim').get_parameter_value().integer_value
#         output_dim = self.get_parameter('model.output_dim').get_parameter_value().integer_value
#         pit_model_weight_path = self.get_parameter('model.pit_model_weight').get_parameter_value().string_value
#         outside_model_weight_path = self.get_parameter('model.outside_model_weight').get_parameter_value().string_value
#         self.pit_exit_x = self.get_parameter('pit_exit.x').get_parameter_value().double_value
#         self.pit_exit_y = self.get_parameter('pit_exit.y').get_parameter_value().double_value
#         self.pit_exit_radius = self.get_parameter('pit_exit.radius').get_parameter_value().double_value
#         self.pit_exit_radius_sq = self.pit_exit_radius ** 2
#         self.pit_accel = self.get_parameter('acceleration.pit').get_parameter_value().double_value
#         self.outside_accel = self.get_parameter('acceleration.outside').get_parameter_value().double_value
        
#         self.get_logger().info(f"Pit exit configured at ({self.pit_exit_x}, {self.pit_exit_y}) with radius {self.pit_exit_radius:.2f}m")

#         # --- デバイス設定 ---
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.get_logger().info(f"Using device: {self.device}")

#         # --- モデルのロード ---
#         self.pit_model = TinyLidarNet(input_dim=self.input_dim, output_dim=output_dim).to(self.device)
#         self.pit_model.load_state_dict(torch.load(pit_model_weight_path, map_location=self.device))
#         self.pit_model.eval()
#         self.get_logger().info(f"Pit model loaded from {pit_model_weight_path}")

#         self.outside_model = TinyLidarNet(input_dim=self.input_dim, output_dim=output_dim).to(self.device)
#         self.outside_model.load_state_dict(torch.load(outside_model_weight_path, map_location=self.device))
#         self.outside_model.eval()
#         self.get_logger().info(f"Outside model loaded from {outside_model_weight_path}")

#         # --- 変数の初期化 ---
#         self.latest_scan_msg = None
#         self.latest_pose_msg = None 
#         self.is_in_pit = True        

#         # --- QoSプロファイル ---
#         sensor_qos = QoSProfile(
#             reliability=ReliabilityPolicy.BEST_EFFORT,
#             history=HistoryPolicy.KEEP_LAST,
#             depth=1
#         )

#         # --- トピック名 ---
#         SCAN_TOPIC = '/scan'
#         CONTROL_TOPIC = '/awsim/control_cmd'
#         POSE_TOPIC = '/localization/pose_with_covariance'
#         MARKER_TOPIC = '/pit_exit_marker'
#         RESET_TOPIC = '/aichallenge/awsim/reset' # ▼▼▼【追加】リセット用トピック名 ▼▼▼

#         # --- 通信設定 ---
#         self.create_subscription(LaserScan, SCAN_TOPIC, self.scan_callback, sensor_qos)
#         self.create_subscription(PoseWithCovarianceStamped, POSE_TOPIC, self.pose_callback, 10)
#         self.create_subscription(Empty, RESET_TOPIC, self.reset_callback, 10)
        
#         self.control_pub = self.create_publisher(AckermannControlCommand, CONTROL_TOPIC, 1)
#         self.timer = self.create_timer(1.0 / timer_hz, self.timer_callback)
#         self.marker_pub = self.create_publisher(Marker, MARKER_TOPIC, 1)
#         self.marker_timer = self.create_timer(1.0, self.publish_pit_exit_marker)
        
#         self.get_logger().info("TinyLidarNetNode has been initialized successfully.")

#     def scan_callback(self, msg):
#         self.latest_scan_msg = msg

#     def pose_callback(self, msg):
#         self.latest_pose_msg = msg

#     def reset_callback(self, msg):
#         """
#         /aichallenge/awsim/reset トピックを受け取った際に状態を初期化する
#         """
#         self.get_logger().warn("<<<<< Received reset signal. Returning to PIT mode. >>>>>")
#         self.is_in_pit = True
#         # リセット時に最新のセンサーデータをクリアし、古いデータでの誤動作を防ぐ
#         self.latest_scan_msg = None
#         self.latest_pose_msg = None

#     def check_pit_exit(self):
#         if not self.is_in_pit or self.latest_pose_msg is None:
#             return

#         current_x = self.latest_pose_msg.pose.pose.position.x
#         current_y = self.latest_pose_msg.pose.pose.position.y
#         dist_sq = (current_x - self.pit_exit_x)**2 + (current_y - self.pit_exit_y)**2
        
#         if dist_sq < self.pit_exit_radius_sq:
#             self.is_in_pit = False
#             self.get_logger().warn(
#                 f"!!! Exited pit area at (x={current_x:.2f}, y={current_y:.2f}). "
#                 f"Switching to OUTSIDE model and changing acceleration. !!!"
#             )

#     def publish_pit_exit_marker(self):
#         marker = Marker()
#         marker.header.frame_id = "map"
#         marker.header.stamp = self.get_clock().now().to_msg()
#         marker.ns = "pit_exit_area"
#         marker.id = 0
#         marker.type = Marker.CYLINDER
#         marker.action = Marker.ADD
#         marker.pose.position.x = self.pit_exit_x
#         marker.pose.position.y = self.pit_exit_y
#         marker.pose.position.z = 0.0
#         marker.pose.orientation.w = 1.0
#         marker.scale.x = self.pit_exit_radius * 2.0
#         marker.scale.y = self.pit_exit_radius * 2.0
#         marker.scale.z = 0.1
#         marker.color.r = 1.0
#         marker.color.g = 0.0
#         marker.color.b = 0.0
#         marker.color.a = 0.4
#         marker.lifetime.sec = 0
#         self.marker_pub.publish(marker)

#     def timer_callback(self):
#         if self.latest_scan_msg is None or self.latest_pose_msg is None:
#             if self.latest_scan_msg is None:
#                 self.get_logger().warn("Waiting for LaserScan data...", throttle_duration_sec=5)
#             if self.latest_pose_msg is None:
#                 self.get_logger().warn("Waiting for Pose data...", throttle_duration_sec=5)
#             return

#         self.check_pit_exit()

#         try:
#             ranges = np.array(self.latest_scan_msg.ranges, dtype=np.float32)

#             max_range = self.latest_scan_msg.range_max
#             ranges[np.isinf(ranges)] = max_range
#             ranges[np.isnan(ranges)] = 0.0
            
#             current_len = len(ranges)
#             if current_len > self.input_dim:
#                 start_index = (current_len - self.input_dim) // 2
#                 processed_ranges = ranges[start_index : start_index + self.input_dim]
#             elif current_len < self.input_dim:
#                 pad_width = self.input_dim - current_len
#                 processed_ranges = np.pad(ranges, (0, pad_width), 'constant', constant_values=0)
#             else:
#                 processed_ranges = ranges

#             processed_ranges /= 30.0
#             scan_tensor = torch.from_numpy(processed_ranges).unsqueeze(0).to(self.device)
#         except Exception as e:
#             self.get_logger().error(f"Data preprocessing failed: {e}")
#             return
            
#         active_model = self.pit_model if self.is_in_pit else self.outside_model
        
#         with torch.no_grad():
#             if scan_tensor.dim() == 2:
#                 scan_tensor = scan_tensor.unsqueeze(1)
#             outputs = active_model(scan_tensor)
        
#         predicted_steering_angle = outputs[0, 1].item()
        
#         control_msg = AckermannControlCommand()
#         control_msg.stamp = self.get_clock().now().to_msg()
        
#         current_accel = self.pit_accel if self.is_in_pit else self.outside_accel
        
#         control_msg.longitudinal.acceleration = current_accel
#         control_msg.lateral.steering_tire_angle = predicted_steering_angle
        
#         self.control_pub.publish(control_msg)
        
#         mode_str = "PIT" if self.is_in_pit else "OUTSIDE"
#         self.get_logger().info(
#             f"[{mode_str}] Published: Steer={predicted_steering_angle:.3f} (Accel CMD: {current_accel})",
#             throttle_duration_sec=1.0
#         )
        
# def main(args=None):
#     rclpy.init(args=args)
#     node = TinyLidarNetNode()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()

# if __name__ == '__main__':
#     main()


import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np
from numpy.lib.stride_tricks import as_strided

import math

# --- ROS2メッセージの型をインポート ---
from sensor_msgs.msg import LaserScan
from autoware_auto_control_msgs.msg import AckermannControlCommand
from geometry_msgs.msg import PoseWithCovarianceStamped
from visualization_msgs.msg import Marker
from std_msgs.msg import Empty

# ▼▼▼【ここから NumPyによるモデル実装】▼▼▼
# PyTorchの代わりに、NumPyでニューラルネットワークの計算を再現します。

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

def linear(x, weight, bias):
    return np.dot(x, weight.T) + bias

def conv1d(x, weight, bias, stride):
    """
    forループを使わず、stride_tricksで高速化した1次元畳み込み。
    x:      入力 (バッチサイズ, 入力チャネル, 入力長)
    weight: 重み (出力チャネル, 入力チャネル, カーネルサイズ)
    bias:   バイアス (出力チャネル)
    stride: ストライド
    """
    # --- 形状に関する情報を取得 ---
    n_x, c_in, l_in = x.shape
    c_out, _, k = weight.shape
    l_out = (l_in - k) // stride + 1

    # --- stride_tricks を使って畳み込みビューを作成 ---
    # 入力xの各次元を1つ進むのに必要なバイト数を取得
    s0, s1, s2 = x.strides
    
    # メモリをコピーすることなく、畳み込みの窓がスライドしていく様子を表現した
    # 4次元の「ビュー（view）」を作成する
    strided_x = as_strided(x,
                           shape=(n_x, c_in, l_out, k),
                           strides=(s0, s1, s2 * stride, s2))

    # --- 畳み込み計算を行列積として一括実行 ---
    # einsumを使い、ビューと重み行列の内積を計算して畳み込みを完了させる
    # 'nclk,ock->nol' は以下の計算を指定:
    # n: バッチ, c: 入力チャネル, l: 出力長, k: カーネルサイズ, o: 出力チャネル
    # cとkの次元で和を取り、n, o, l の次元を残す
    conv_val = np.einsum('nclk,ock->nol', strided_x, weight)
    
    # --- バイアスを加算して結果を返す ---
    # ブロードキャストが正しく行われるようにバイアスの形状を調整
    return conv_val + bias.reshape(1, -1, 1)


class NumpyTinyLidarNet:
    """ TinyLidarNetモデルの推論をNumPyで実行するクラス """
    def __init__(self, weights_path):
        w = np.load(weights_path)
        self.conv1_w, self.conv1_b = w['conv1.weight'], w['conv1.bias']
        self.conv2_w, self.conv2_b = w['conv2.weight'], w['conv2.bias']
        self.conv3_w, self.conv3_b = w['conv3.weight'], w['conv3.bias']
        self.conv4_w, self.conv4_b = w['conv4.weight'], w['conv4.bias']
        self.conv5_w, self.conv5_b = w['conv5.weight'], w['conv5.bias']
        self.fc1_w, self.fc1_b = w['fc1.weight'], w['fc1.bias']
        self.fc2_w, self.fc2_b = w['fc2.weight'], w['fc2.bias']
        self.fc3_w, self.fc3_b = w['fc3.weight'], w['fc3.bias']
        self.fc4_w, self.fc4_b = w['fc4.weight'], w['fc4.bias']

    def forward(self, x):
        # このforwardパスは変更なし。呼び出すconv1dが高速化されたバージョンになる。
        x = np.expand_dims(x, axis=1) # (B, L) -> (B, 1, L)
        x = relu(conv1d(x, self.conv1_w, self.conv1_b, stride=4))
        x = relu(conv1d(x, self.conv2_w, self.conv2_b, stride=4))
        x = relu(conv1d(x, self.conv3_w, self.conv3_b, stride=2))
        x = relu(conv1d(x, self.conv4_w, self.conv4_b, stride=1))
        x = relu(conv1d(x, self.conv5_w, self.conv5_b, stride=1))
        x = x.reshape(x.shape[0], -1) # Flatten
        x = relu(linear(x, self.fc1_w, self.fc1_b))
        x = relu(linear(x, self.fc2_w, self.fc2_b))
        x = relu(linear(x, self.fc3_w, self.fc3_b))
        x = tanh(linear(x, self.fc4_w, self.fc4_b))
        return x
# ▲▲▲【ここまで NumPyによるモデル実装】▲▲▲

class TinyLidarNetNode(Node):
    def __init__(self):
        super().__init__('tiny_lidar_net_node')

        # --- パラメータの宣言 (重みファイルの拡張子を.npzに変更) ---
        self.declare_parameter('timer_hz', 50.0)
        self.declare_parameter('model.input_dim', 1080)
        # self.declare_parameter('model.output_dim', 2) # output_dimはモデル内部で決定されるため不要
        self.declare_parameter('model.pit_model_weight', 'pit_model_weights.npz') # <- 変更
        self.declare_parameter('model.outside_model_weight', 'outside_model_weights.npz') # <- 変更
        self.declare_parameter('pit_exit.x', 0.0)
        self.declare_parameter('pit_exit.y', 0.0)
        self.declare_parameter('pit_exit.radius', 10.0)
        self.declare_parameter('acceleration.pit', 1.0)
        self.declare_parameter('acceleration.outside', 0.0001)

        # --- パラメータの取得 ---
        timer_hz = self.get_parameter('timer_hz').get_parameter_value().double_value
        self.input_dim = self.get_parameter('model.input_dim').get_parameter_value().integer_value
        pit_model_weight_path = self.get_parameter('model.pit_model_weight').get_parameter_value().string_value
        outside_model_weight_path = self.get_parameter('model.outside_model_weight').get_parameter_value().string_value
        self.pit_exit_x = self.get_parameter('pit_exit.x').get_parameter_value().double_value
        self.pit_exit_y = self.get_parameter('pit_exit.y').get_parameter_value().double_value
        self.pit_exit_radius = self.get_parameter('pit_exit.radius').get_parameter_value().double_value
        self.pit_exit_radius_sq = self.pit_exit_radius ** 2
        self.pit_accel = self.get_parameter('acceleration.pit').get_parameter_value().double_value
        self.outside_accel = self.get_parameter('acceleration.outside').get_parameter_value().double_value
        
        self.get_logger().info(f"Pit exit configured at ({self.pit_exit_x}, {self.pit_exit_y}) with radius {self.pit_exit_radius:.2f}m")

        # --- 【変更点】モデルのロード (NumPy版) ---
        self.get_logger().info("Using NumPy for model inference.")
        try:
            self.pit_model = NumpyTinyLidarNet(weights_path=pit_model_weight_path)
            self.get_logger().info(f"Pit model loaded from {pit_model_weight_path}")

            self.outside_model = NumpyTinyLidarNet(weights_path=outside_model_weight_path)
            self.get_logger().info(f"Outside model loaded from {outside_model_weight_path}")
        except FileNotFoundError as e:
            self.get_logger().fatal(f"Weight file not found: {e}. Please make sure to convert .pth to .npz and provide the correct path.")
            # エラー発生時はノードを終了
            rclpy.shutdown()
            return
            
        # --- 変数の初期化 ---
        self.latest_scan_msg = None
        self.latest_pose_msg = None 
        self.is_in_pit = True        

        # --- QoSプロファイル ---
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # --- トピック名 ---
        SCAN_TOPIC = '/scan'
        CONTROL_TOPIC = '/awsim/control_cmd'
        POSE_TOPIC = '/localization/pose_with_covariance'
        MARKER_TOPIC = '/pit_exit_marker'
        RESET_TOPIC = '/aichallenge/awsim/reset'

        # --- 通信設定 ---
        self.create_subscription(LaserScan, SCAN_TOPIC, self.scan_callback, sensor_qos)
        self.create_subscription(PoseWithCovarianceStamped, POSE_TOPIC, self.pose_callback, 10)
        self.create_subscription(Empty, RESET_TOPIC, self.reset_callback, 10)
        
        self.control_pub = self.create_publisher(AckermannControlCommand, CONTROL_TOPIC, 1)
        self.timer = self.create_timer(1.0 / timer_hz, self.timer_callback)
        self.marker_pub = self.create_publisher(Marker, MARKER_TOPIC, 1)
        self.marker_timer = self.create_timer(1.0, self.publish_pit_exit_marker)
        
        self.get_logger().info("TinyLidarNetNode (NumPy version) has been initialized successfully.")

    # (scan_callback, pose_callback, reset_callback, check_pit_exit, publish_pit_exit_marker の各メソッドは変更なし)
    def scan_callback(self, msg):
        self.latest_scan_msg = msg

    def pose_callback(self, msg):
        self.latest_pose_msg = msg

    def reset_callback(self, msg):
        self.get_logger().warn("<<<<< Received reset signal. Returning to PIT mode. >>>>>")
        self.is_in_pit = True
        self.latest_scan_msg = None
        self.latest_pose_msg = None

    def check_pit_exit(self):
        if not self.is_in_pit or self.latest_pose_msg is None:
            return
        current_x = self.latest_pose_msg.pose.pose.position.x
        current_y = self.latest_pose_msg.pose.pose.position.y
        dist_sq = (current_x - self.pit_exit_x)**2 + (current_y - self.pit_exit_y)**2
        if dist_sq < self.pit_exit_radius_sq: # ピットの外に出たら
            self.is_in_pit = False
            self.get_logger().warn(
                f"!!! Exited pit area at (x={current_x:.2f}, y={current_y:.2f}). "
                f"Switching to OUTSIDE model and changing acceleration. !!!"
            )

    def publish_pit_exit_marker(self):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "pit_exit_area"
        marker.id = 0
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.pose.position.x = self.pit_exit_x
        marker.pose.position.y = self.pit_exit_y
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = self.pit_exit_radius * 2.0
        marker.scale.y = self.pit_exit_radius * 2.0
        marker.scale.z = 0.1
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.4
        marker.lifetime.sec = 0 # 永続表示
        self.marker_pub.publish(marker)


    def timer_callback(self):
        if self.latest_scan_msg is None or self.latest_pose_msg is None:
            # (ログ出力部分は変更なし)
            return

        self.check_pit_exit()

        # --- 【変更点】データ前処理と推論 (NumPy版) ---
        try:
            # 1. センサーデータの前処理 (NumPyのまま)
            ranges = np.array(self.latest_scan_msg.ranges, dtype=np.float32)
            max_range = self.latest_scan_msg.range_max
            ranges[np.isinf(ranges)] = max_range
            ranges[np.isnan(ranges)] = 0.0
            
            current_len = len(ranges)
            if current_len > self.input_dim:
                start_index = (current_len - self.input_dim) // 2
                processed_ranges = ranges[start_index : start_index + self.input_dim]
            elif current_len < self.input_dim:
                pad_width = self.input_dim - current_len
                processed_ranges = np.pad(ranges, (0, pad_width), 'constant', constant_values=0)
            else:
                processed_ranges = ranges
            
            processed_ranges /= 30.0
            
            # 2. NumPy配列をモデルの入力形式に変換
            scan_array = np.expand_dims(processed_ranges, axis=0) # (1, 1080)

        except Exception as e:
            self.get_logger().error(f"Data preprocessing failed: {e}")
            return
        
        # 3. アクティブなモデルを選択し、NumPyで推論
        active_model = self.pit_model if self.is_in_pit else self.outside_model
        outputs = active_model.forward(scan_array)
        
        # 4. NumPy配列から結果を取得
        predicted_steering_angle = float(outputs[0, 1])
        
        # --- 制御コマンドの送信 (変更なし) ---
        control_msg = AckermannControlCommand()
        control_msg.stamp = self.get_clock().now().to_msg()
        current_accel = self.pit_accel if self.is_in_pit else self.outside_accel
        control_msg.longitudinal.acceleration = current_accel
        control_msg.lateral.steering_tire_angle = predicted_steering_angle
        self.control_pub.publish(control_msg)
        
        mode_str = "PIT" if self.is_in_pit else "OUTSIDE"
        self.get_logger().info(
            f"[{mode_str}] Published: Steer={predicted_steering_angle:.3f} (Accel CMD: {current_accel})",
            throttle_duration_sec=1.0
        )

# (main関数は変更なし)
def main(args=None):
    rclpy.init(args=args)
    node = TinyLidarNetNode()
    if rclpy.ok():
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()