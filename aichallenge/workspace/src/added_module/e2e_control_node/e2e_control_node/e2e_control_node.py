import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy # <- 変更点 1: QoS関連のクラスをインポート
from sensor_msgs.msg import Image
from autoware_auto_control_msgs.msg import AckermannControlCommand

import torch
from cv_bridge import CvBridge
import numpy as np

# 以前作成したモジュールをインポート
from .src.model.net import DrivingModel
from .src.data.transform import get_transforms

class InferenceNode(Node):
    def __init__(self):
        super().__init__('driving_model_inference_node')
        
        # ROS 2 パラメータの宣言
        self.declare_parameter('model_path', '')
        self.declare_parameter('device', 'cuda')

        # パラメータの取得
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.device = torch.device(self.get_parameter('device').get_parameter_value().string_value)
        
        if not model_path:
            self.get_logger().fatal("モデルのパスが指定されていません。'--ros-args -p model_path:=/path/to/model.pth' を付けて実行してください。")
            rclpy.shutdown()
            return

        self.get_logger().info(f"Loading model from: {model_path}")
        self.get_logger().info(f"Using device: {self.device}")

        # モデルの読み込み
        self.model = DrivingModel(model_name='resnet18').to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()  # モデルを評価モードに設定
        
        # 推論用のTransformを取得 (データ拡張なし)
        self.transform = get_transforms()['val']
        
        # CV Bridgeの初期化
        self.bridge = CvBridge()
        
        # SubscriberとPublisherの作成

        # <- 変更点 2: BEST_EFFORTのQoSプロファイルを作成
        image_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=1 # センサーデータではdepth=1もよく使われる
        )
        
        self.subscription = self.create_subscription(
            Image,
            '/sensing/camera/image_raw', # 購読するトピック名
            self.image_callback,
            image_qos_profile) 

        self.publisher = self.create_publisher(
            AckermannControlCommand,
            '/awsim/control_cmd', # 配信するトピック名
            10)

        self.get_logger().info('Inference node has been initialized.')

    def image_callback(self, msg: Image):
        
        try:
            # ROS ImageメッセージをOpenCV画像に変換
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            
            # 画像の前処理
            sample = {'image': cv_image, 'command': None} 
            transformed_sample = self.transform(sample)
            image_tensor = transformed_sample['image']
            
            # バッチ次元を追加してデバイスに転送
            input_tensor = image_tensor.unsqueeze(0).to(self.device)
            
            # 推論の実行
            with torch.no_grad():
                outputs = self.model(input_tensor)
                
                # 推論結果をCPUに戻し、Pythonの数値に変換
                accel = outputs['accel'].cpu().item()
                steer = outputs['steer'].cpu().item()
            
            # AckermannControlCommandメッセージを作成して配信
            cmd_msg = AckermannControlCommand()
            cmd_msg.stamp = self.get_clock().now().to_msg()
            # 推論結果をメッセージにセット
            cmd_msg.longitudinal.speed = 0.0
            cmd_msg.longitudinal.acceleration = float(accel)
            cmd_msg.lateral.steering_tire_angle = float(steer)
            
            self.publisher.publish(cmd_msg)
            
        except Exception as e:
            self.get_logger().error(f'Failed to process image: {e}')

def main(args=None):
    rclpy.init(args=args)
    try:
        inference_node = InferenceNode()
        rclpy.spin(inference_node)
    except KeyboardInterrupt:
        pass
    finally:
        inference_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()