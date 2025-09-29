import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np
from numpy.lib.stride_tricks import as_strided
import time

from sensor_msgs.msg import LaserScan
from autoware_auto_control_msgs.msg import AckermannControlCommand
from std_msgs.msg import Empty

# --- NumPyによるニューラルネットワーク実装 ---
def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

def linear(x, weight, bias):
    return np.dot(x, weight.T) + bias

def conv1d(x, weight, bias, stride):
    n_x, c_in, l_in = x.shape
    c_out, _, k = weight.shape
    l_out = (l_in - k) // stride + 1
    s0, s1, s2 = x.strides
    strided_x = as_strided(x, shape=(n_x, c_in, l_out, k), strides=(s0, s1, s2 * stride, s2))
    strided_x_reshaped = strided_x.transpose(0, 2, 1, 3).reshape(n_x * l_out, c_in * k)
    weight_reshaped = weight.reshape(c_out, -1)
    conv_val = strided_x_reshaped @ weight_reshaped.T
    conv_val_reshaped = conv_val.reshape(n_x, l_out, c_out).transpose(0, 2, 1)
    return conv_val_reshaped + bias.reshape(1, -1, 1)

class NumpyTinyLidarNet:
    """ ★★★ 従来モデル ★★★ """
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
        x = np.expand_dims(x, axis=1)
        x = relu(conv1d(x, self.conv1_w, self.conv1_b, stride=4))
        x = relu(conv1d(x, self.conv2_w, self.conv2_b, stride=4))
        x = relu(conv1d(x, self.conv3_w, self.conv3_b, stride=2))
        x = relu(conv1d(x, self.conv4_w, self.conv4_b, stride=1))
        x = relu(conv1d(x, self.conv5_w, self.conv5_b, stride=1))
        x = x.reshape(x.shape[0], -1)
        x = relu(linear(x, self.fc1_w, self.fc1_b))
        x = relu(linear(x, self.fc2_w, self.fc2_b))
        x = relu(linear(x, self.fc3_w, self.fc3_b))
        x = tanh(linear(x, self.fc4_w, self.fc4_b))
        return x

class NumpyTinyLidarNetSmall:
    """ ★★★ 軽量版モデルクラス ★★★ """
    def __init__(self, weights_path):
        w = np.load(weights_path)
        self.conv1_w, self.conv1_b = w['conv1.weight'], w['conv1.bias']
        self.conv2_w, self.conv2_b = w['conv2.weight'], w['conv2.bias']
        self.conv3_w, self.conv3_b = w['conv3.weight'], w['conv3.bias']
        self.fc1_w, self.fc1_b = w['fc1.weight'], w['fc1.bias']
        self.fc2_w, self.fc2_b = w['fc2.weight'], w['fc2.bias']
        self.fc3_w, self.fc3_b = w['fc3.weight'], w['fc3.bias']

    def forward(self, x):
        x = np.expand_dims(x, axis=1)
        x = relu(conv1d(x, self.conv1_w, self.conv1_b, stride=4))
        x = relu(conv1d(x, self.conv2_w, self.conv2_b, stride=4))
        x = relu(conv1d(x, self.conv3_w, self.conv3_b, stride=2))
        x = x.reshape(x.shape[0], -1)
        x = relu(linear(x, self.fc1_w, self.fc1_b))
        x = relu(linear(x, self.fc2_w, self.fc2_b))
        x = tanh(linear(x, self.fc3_w, self.fc3_b))
        return x


class E2ENode(Node):
    def __init__(self):
        super().__init__('e2e_node_py')

        # === パラメータ宣言 (簡略化) ===
        self.declare_parameter('log_interval_sec', 5.0)
        self.declare_parameter('model.input_dim', 1080)
        self.declare_parameter('model.architecture', 'large') 
        self.declare_parameter('model.weight_path', '') 
        self.declare_parameter('acceleration', 0.1) 
        
        # === パラメータ取得 (簡略化) ===
        self.log_interval = self.get_parameter('log_interval_sec').get_parameter_value().double_value
        self.input_dim = self.get_parameter('model.input_dim').get_parameter_value().integer_value
        self.model_architecture = self.get_parameter('model.architecture').get_parameter_value().string_value 
        model_weight_path = self.get_parameter('model.weight_path').get_parameter_value().string_value
        self.acceleration = self.get_parameter('acceleration').get_parameter_value().double_value

        self.get_logger().info(f"Model architecture is set to: '{self.model_architecture}'") 

        # === モデルのロード (単一化) ===
        ModelClass = None
        if self.model_architecture == 'small':
            ModelClass = NumpyTinyLidarNetSmall
        else: # 'large' またはその他の場合はデフォルト
            ModelClass = NumpyTinyLidarNet

        if not model_weight_path:
            self.get_logger().fatal("Model weight path is not set.")
            rclpy.shutdown(); return
        try:
            self.model = ModelClass(weights_path=model_weight_path)
            self.get_logger().info(f"Model ({self.model_architecture}) loaded from {model_weight_path}")
        except Exception as e:
            self.get_logger().fatal(f"Model file not found or structure mismatch: {e}")
            rclpy.shutdown(); return

        # === 変数の初期化 (簡略化) ===
        self.inference_times = []
        self.last_log_time = self.get_clock().now()

        # === 通信設定 (簡略化) ===
        sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(LaserScan, "/scan", self.scan_callback, sensor_qos)
        self.create_subscription(Empty, "/aichallenge/awsim/reset", self.reset_callback, 10)
        self.control_pub = self.create_publisher(AckermannControlCommand, "/awsim/control_cmd", 1)
        
        self.get_logger().info("E2E Python Node (Simplified) has been initialized successfully.")
        
    def scan_callback(self, msg):
        start_time = time.monotonic()
        
        # === 常にAIモデルで推論 (分岐を削除) ===
        ranges = np.array(msg.ranges, dtype=np.float32)
        ranges[np.isinf(ranges)] = msg.range_max
        ranges[np.isnan(ranges)] = 0.0
        
        mode_str = f"AI-{self.model_architecture}"
        processed_ranges = self.process_ranges_for_model(ranges)
        scan_array = np.expand_dims(processed_ranges, axis=0)
        
        # 単一モデルでフォワードパスを実行
        outputs = self.model.forward(scan_array)
        steering = float(outputs[0, 1])
        
        steering = np.clip(steering, -1.0, 1.0)
        
        # 推論時間計測
        end_time = time.monotonic()
        duration_ms = (end_time - start_time) * 1000.0
        self.inference_times.append(duration_ms)
        
        # 制御コマンド生成
        cmd = AckermannControlCommand()
        cmd.stamp = self.get_clock().now().to_msg()
        cmd.longitudinal.acceleration = self.acceleration # 固定の加速度を使用
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
        # (変更なし)
        if len(ranges) > self.input_dim:
            indices = np.linspace(0, len(ranges) - 1, self.input_dim, dtype=int)
            processed = ranges[indices]
        elif len(ranges) < self.input_dim:
            pad_width = self.input_dim - len(ranges)
            processed = np.pad(ranges, (0, pad_width), 'constant', constant_values=0)
        else:
            processed = ranges
        processed /= 30.0
        return processed

    def reset_callback(self, msg):
        self.get_logger().warn("<<<<< Received reset signal. Clearing inference stats. >>>>>")
        # リセット時の処理を簡略化
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