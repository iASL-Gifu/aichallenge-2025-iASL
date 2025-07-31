#include "image_process_node.hpp"

ImageProcessNode::ImageProcessNode() : Node("image_process_node")
{
  // パラメータの宣言と取得
  this->declare_parameter<double>("output_rate_hz", 30.0);
  this->declare_parameter<int>("output_width", 640);
  this->declare_parameter<int>("output_height", 480);

  this->get_parameter("output_rate_hz", output_rate_hz_);
  this->get_parameter("output_width", output_width_);
  this->get_parameter("output_height", output_height_);

  RCLCPP_INFO(this->get_logger(), "Output rate: %.1f Hz", output_rate_hz_);
  RCLCPP_INFO(this->get_logger(), "Output size: %d x %d", output_width_, output_height_);
  
  // QoSプロファイルの設定 (Best Effort)
  auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort();

  // ImageTransportを使用してサブスクライバーとパブリッシャーを作成
  image_transport::ImageTransport it(shared_from_this());
  sub_ = it.subscribe("input_image", 1, std::bind(&ImageProcessNode::image_callback, this, std::placeholders::_1), "raw", qos.get_rmw_qos_profile());
  pub_ = it.advertise("output_image", 1);
  
  // 指定した周波数でタイマーを作成
  auto timer_period = std::chrono::duration<double>(1.0 / output_rate_hz_);
  timer_ = this->create_wall_timer(timer_period, std::bind(&ImageProcessNode::timer_callback, this));
}

// 画像を受信したときのコールバック関数
void ImageProcessNode::image_callback(const sensor_msgs::msg::Image::ConstSharedPtr& msg)
{
  std::lock_guard<std::mutex> lock(image_mutex_);
  latest_image_msg_ = msg; // 最新の画像を格納
}

// タイマーで周期的に呼ばれる関数
void ImageProcessNode::timer_callback()
{
  sensor_msgs::msg::Image::ConstSharedPtr current_image_msg;

  {
    std::lock_guard<std::mutex> lock(image_mutex_);
    if (!latest_image_msg_) {
      return;
    }
    current_image_msg = latest_image_msg_;
  }

  // cv_bridgeを使用してROS ImageメッセージをOpenCVのMat形式に変換
  cv_bridge::CvImagePtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvCopy(current_image_msg, sensor_msgs::image_encodings::BGR8);
  } catch (cv_bridge::Exception& e) {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    return;
  }

  // 画像をリサイズ
  cv::Mat resized_image;
  cv::resize(cv_ptr->image, resized_image, cv::Size(output_width_, output_height_), 0, 0, cv::INTER_AREA);

  sensor_msgs::msg::Image::SharedPtr output_msg = cv_bridge::CvImage(
    cv_ptr->header, // 元画像のヘッダーを使用
    sensor_msgs::image_encodings::BGR8, 
    resized_image
  ).toImageMsg();

  // ヘッダーのタイムスタンプを現在時刻に更新
  output_msg->header.stamp = this->now();

  // 画像をパブリッシュ
  pub_.publish(output_msg);
}

// メイン関数
int main(int argc, char* argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ImageProcessNode>());
  rclcpp::shutdown();
  return 0;
}