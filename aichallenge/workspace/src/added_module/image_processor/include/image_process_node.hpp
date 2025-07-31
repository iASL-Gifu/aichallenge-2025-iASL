#ifndef IMAGE_PROCESS_NODE_HPP_
#define IMAGE_PROCESS_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.hpp>
#include <opencv2/opencv.hpp>
#include <mutex>

class ImageProcessNode : public rclcpp::Node
{
public:
  ImageProcessNode();

private:
  // コールバック関数
  void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr& msg);
  void timer_callback();

  // ROS関連のメンバ変数
  image_transport::Subscriber sub_;
  image_transport::Publisher pub_;
  rclcpp::TimerBase::SharedPtr timer_;

  // 最後に受信した画像メッセージを保持
  sensor_msgs::msg::Image::ConstSharedPtr latest_image_msg_;
  std::mutex image_mutex_; // latest_image_msg_へのアクセスを保護するミューテックス

  // パラメータ
  double output_rate_hz_;
  int output_width_;
  int output_height_;
};

#endif // IMAGE_PROCESS_NODE_HPP_