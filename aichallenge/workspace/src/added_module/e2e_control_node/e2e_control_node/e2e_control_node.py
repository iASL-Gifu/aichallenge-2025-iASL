import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from autoware_auto_control_msgs.msg import AckermannControlCommand

import torch
from cv_bridge import CvBridge
import numpy as np
import threading 

# 以前作成したモジュールをインポート
from .src.model.net import DrivingModel
from .src.data.transform import get_transforms

class InferenceNode(Node):
    def __init__(self):
        super().__init__('driving_model_inference_node')
        
        # ROS 2 パラメータの宣言
        self.declare_parameter('model_path', '')
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('inference_hz', 30.0) 

        # パラメータの取得
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.device = torch.device(self.get_parameter('device').get_parameter_value().string_value)
        inference_hz = self.get_parameter('inference_hz').get_parameter_value().double_value

        if not model_path:
            self.get_logger().fatal("モデルのパスが指定されていません。'--ros-args -p model_path:=/path/to/model.pth' を付けて実行してください。")
            rclpy.shutdown()
            return

        self.get_logger().info(f"Loading model from: {model_path}")
        self.get_logger().info(f"Using device: {self.device}")
        self.get_logger().info(f"Inference frequency set to: {inference_hz} Hz")

        # モデルの読み込み
        self.model = DrivingModel(model_name='resnet18').to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        self.transform = get_transforms()['val']
        self.bridge = CvBridge()

        ### <--- 変更点 3: 最新の画像とロックを保持する変数を初期化
        self.latest_cv_image = None
        self.image_lock = threading.Lock()
        
        # Subscriber
        image_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=1
        )
        self.subscription = self.create_subscription(
            Image,
            '/sensing/camera/image_raw',
            self.image_callback, # 画像を保存するコールバック
            image_qos_profile) 

        # Publisher
        self.publisher = self.create_publisher(
            AckermannControlCommand,
            '/awsim/control_cmd',
            10)

        ### <--- 変更点 4: 指定した周波数で推論を実行するタイマーを作成
        self.timer = self.create_timer(1.0 / inference_hz, self.timer_callback)

        self.get_logger().info('Inference node has been initialized.')

    def image_callback(self, msg: Image):
        """
        画像メッセージを受信し、スレッドセーフに最新の画像として保持する。
        """
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            # ロックをかけて、timer_callbackとの競合を防ぐ
            with self.image_lock:
                self.latest_cv_image = cv_image[:, :, ::-1].copy() # BGR to RGB
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')
    
    def timer_callback(self):
        """
        タイマーによって定期的に呼び出され、推論とコマンド発行を行う。
        """
        # まだ画像を受信していなければ何もしない
        if self.latest_cv_image is None:
            return
            
        # スレッドセーフに画像データのコピーを取得
        with self.image_lock:
            # 推論中に画像が更新されても影響がないように、ローカル変数にコピーする
            cv_image = self.latest_cv_image.copy()

        try:
            # 画像の前処理
            sample = {'image': cv_image, 'command': None} 
            transformed_sample = self.transform(sample)
            image_tensor = transformed_sample['image']
            
            # バッチ次元を追加してデバイスに転送
            input_tensor = image_tensor.unsqueeze(0).to(self.device)
            
            # 推論の実行
            with torch.no_grad():
                outputs = self.model(input_tensor)
                accel = outputs['accel'].cpu().item()
                steer = outputs['steer'].cpu().item()
            
            # AckermannControlCommandメッセージを作成して配信
            cmd_msg = AckermannControlCommand()
            cmd_msg.stamp = self.get_clock().now().to_msg()
            cmd_msg.longitudinal.speed = 0.0
            cmd_msg.longitudinal.acceleration = float(accel)
            cmd_msg.lateral.steering_tire_angle = float(steer)
            
            self.publisher.publish(cmd_msg)
            
        except Exception as e:
            self.get_logger().error(f'Failed to process image in timer callback: {e}')


def main(args=None):
    rclpy.init(args=args)
    try:
        inference_node = InferenceNode()
        executor = rclpy.executors.MultiThreadedExecutor()
        executor.add_node(inference_node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if 'inference_node' in locals() and rclpy.ok():
            inference_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()