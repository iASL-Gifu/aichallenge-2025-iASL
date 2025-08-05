import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image as ImageMsg
from sensor_msgs.msg import Imu as ImuMsg
from autoware_auto_control_msgs.msg import AckermannControlCommand

import torch
from cv_bridge import CvBridge
import cv2
import numpy as np
import threading
from PIL import Image
import time

from src.model.dual_net import ModelA_StateFixed, ModelB_StateUpdating, ModelC_HybridSkipConnection
from src.data.transform import val_test_transform

MODEL_MAP = {
    "ModelA_StateFixed": ModelA_StateFixed,
    "ModelB_StateUpdating": ModelB_StateUpdating,
    "ModelC_HybridSkipConnection": ModelC_HybridSkipConnection,
}

class DrivingInferenceNode(Node):
    def __init__(self):
        super().__init__('driving_model_inference_node')
        
        self.declare_parameter('model_path', '')
        self.declare_parameter('model_name', 'ModelC_HybridSkipConnection')
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('imu_buffer_size', 10)
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        model_name = self.get_parameter('model_name').get_parameter_value().string_value
        self.device = torch.device(self.get_parameter('device').get_parameter_value().string_value)
        self.imu_buffer_size = self.get_parameter('imu_buffer_size').get_parameter_value().integer_value
        
        if not model_path or model_name not in MODEL_MAP:
            self.get_logger().fatal("モデルパスまたはモデル名が無効です。")
            rclpy.shutdown(); return
        
        self.get_logger().info(f"モデルをロード中: {model_path}")
        model_class = MODEL_MAP.get(model_name)
        self.model = model_class()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.bridge = CvBridge()
        self.transform = val_test_transform
        
        self.imu_buffer = []
        self.latest_image_pil = None
        self.data_lock = threading.Lock() # データの受け渡しのみを保護する軽量なロック
        self.plan_ready_flag = False

        sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.image_sub = self.create_subscription(ImageMsg, '/sensing/camera/image_raw', self.image_callback, sensor_qos)
        self.imu_sub = self.create_subscription(ImuMsg, '/sensing/imu/imu_raw', self.imu_callback, sensor_qos)
        self.control_pub = self.create_publisher(AckermannControlCommand, '/awsim/control_cmd', 10)

        self.get_logger().info('推論ノードの初期化が完了しました。最初の画像とIMUデータを待機中...')

    def image_callback(self, msg: ImageMsg):
        """画像を受け取り、最新の画像を保持する"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            image_pil = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            
            # ロックを使って最新の画像をアトミックに更新
            with self.data_lock:
                self.latest_image_pil = image_pil
        except Exception as e:
            self.get_logger().error(f'画像処理中にエラーが発生: {e}')
    
    def imu_callback(self, msg: ImuMsg):
        """すべての推論ロジックをここに集約"""
        
        # 1. IMUデータをバッファリング
        imu_sample = [
            msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z,
            msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z
        ]
        # self.imu_bufferはimu_callbackからしか書き込まれないのでロック不要
        self.imu_buffer.append(imu_sample)
        if len(self.imu_buffer) > self.imu_buffer_size:
            self.imu_buffer.pop(0)

        # 2. 新しい画像があるかチェックし、あればプランを更新
        new_image_to_process = None
        with self.data_lock:
            if self.latest_image_pil is not None:
                new_image_to_process = self.latest_image_pil
                self.latest_image_pil = None 

        if new_image_to_process:
            self.update_driving_plan(new_image_to_process)

        # 3. プランが準備完了なら、制御コマンドを予測してPublish
        if self.plan_ready_flag:
            try:
                imu_tensor = torch.tensor(imu_sample, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
                
                accel, steer = self.model.predict_control(imu_tensor)

                cmd_msg = AckermannControlCommand()
                cmd_msg.stamp = self.get_clock().now().to_msg()
                cmd_msg.longitudinal.speed = 0.0
                cmd_msg.longitudinal.acceleration = float(accel)
                cmd_msg.lateral.steering_tire_angle = float(steer)
                self.control_pub.publish(cmd_msg)
            except Exception as e:
                self.get_logger().error(f'制御予測中にエラーが発生: {e}')


    def update_driving_plan(self, image_pil):
        """運転方針（モデルの内部状態）を更新する。imu_callbackからのみ呼ばれる。"""
        try:
            start_time = time.time()
            self.get_logger().info('新しい画像を検出。運転プランを更新します...')
            
            image_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
            imu_sequence_np = np.array(self.imu_buffer)
            imu_sequence_tensor = torch.from_numpy(imu_sequence_np).float().unsqueeze(0).to(self.device)
            
            self.model.update_plan(image_tensor, imu_sequence_tensor)
            
            if not self.plan_ready_flag:
                self.plan_ready_flag = True
                self.get_logger().info('最初の運転プランが生成されました。推論を開始します。')
            
            elapsed_time = (time.time() - start_time) * 1000
            self.get_logger().info(f'運転方針の更新が完了しました (処理時間: {elapsed_time:.2f} ms)')
        except Exception as e:
            self.get_logger().error(f'プラン更新中にエラーが発生: {e}')


def main(args=None):
    rclpy.init(args=args)
    try:
        inference_node = DrivingInferenceNode()
        executor = MultiThreadedExecutor()
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