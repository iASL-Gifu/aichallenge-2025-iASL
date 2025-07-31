import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from autoware_auto_control_msgs.msg import AckermannControlCommand

import torch
from cv_bridge import CvBridge
import numpy as np
import threading
import time 

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
        self.declare_parameter('model_name', 'resnet18')

        # パラメータの取得
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.device = torch.device(self.get_parameter('device').get_parameter_value().string_value)
        inference_hz = self.get_parameter('inference_hz').get_parameter_value().double_value
        model_name = self.get_parameter('model_name').get_parameter_value().string_value

        if not model_path:
            self.get_logger().fatal("モデルのパスが指定されていません。'--ros-args -p model_path:=/path/to/model.pth' を付けて実行してください。")
            rclpy.shutdown()
            return

        supported_models = ['resnet18', 'resnet34', 'resnet50']
        if model_name not in supported_models:
            self.get_logger().fatal(f"指定されたモデル名 '{model_name}' はサポートされていません。{supported_models} のいずれかを指定してください。")
            rclpy.shutdown()
            return

        self.get_logger().info(f"Loading model from: {model_path}")
        self.get_logger().info(f"Using device: {self.device}")
        self.get_logger().info(f"Inference frequency set to: {inference_hz} Hz")
        self.get_logger().info(f"Using model architecture: {model_name}") # 使用するモデル名もログ出力

        # モデルの読み込み
        self.model = DrivingModel(model_name=model_name).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        self.transform = get_transforms()['val']
        self.bridge = CvBridge()

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
            self.image_callback,
            image_qos_profile) 

        # Publisher
        self.publisher = self.create_publisher(
            AckermannControlCommand,
            '/awsim/control_cmd',
            10)

        self.timer = self.create_timer(1.0 / inference_hz, self.timer_callback)

        self.get_logger().info('Inference node has been initialized.')

    def image_callback(self, msg: Image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            with self.image_lock:
                self.latest_cv_image = cv_image[:, :, ::-1].copy() # BGR to RGB
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')
    
    def timer_callback(self):
        if self.latest_cv_image is None:
            return
            
        with self.image_lock:
            cv_image = self.latest_cv_image.copy()

        try:
            sample = {'image': cv_image, 'command': None} 
            transformed_sample = self.transform(sample)
            image_tensor = transformed_sample['image']
            
            input_tensor = image_tensor.unsqueeze(0).to(self.device)
            
            start_time = time.time()
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                accel = outputs['accel'].cpu().item()
                steer = outputs['steer'].cpu().item()

            end_time = time.time()
            inference_time_ms = (end_time - start_time) * 1000
            self.get_logger().info(f'Inference time: {inference_time_ms:.2f} ms')
            
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