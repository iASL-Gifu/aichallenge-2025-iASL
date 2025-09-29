#include <rclcpp/rclcpp.hpp>
#include <autoware_auto_control_msgs/msg/ackermann_control_command.hpp>
#include <autoware_auto_vehicle_msgs/msg/velocity_report.hpp> 
#include "std_msgs/msg/float32_multi_array.hpp"

#include <deque>
#include <string>
#include <vector>
#include <numeric>
#include <algorithm>
#include <functional>
#include <cmath>
#include "rcl_interfaces/msg/set_parameters_result.hpp"

// トピック名
const char* INPUT_TOPIC = "/input/control_cmd";
const char* OUTPUT_TOPIC = "/output/control_cmd_filtered";
const char* VELOCITY_TOPIC = "/vehicle/status/velocity_status"; 
const char* STATUS_TOPIC = "/status";

class AckermannFilterNode : public rclcpp::Node
{
  using AckermannMsg = autoware_auto_control_msgs::msg::AckermannControlCommand;
  using VelocityMsg = autoware_auto_vehicle_msgs::msg::VelocityReport;
  using StatusMsg = std_msgs::msg::Float32MultiArray;

public:
  AckermannFilterNode()
  : Node("ackermann_filter_node"), current_velocity_mps_(0.0), current_section_(-1.0f)
  {
    // パラメータの宣言
    this->declare_parameter<std::string>("filter_type", "none");
    this->declare_parameter<int>("window_size", 5);
    this->declare_parameter<bool>("use_scale_filter", true);
    this->declare_parameter<std::string>("scale_filter_type", "normal");
    this->declare_parameter<double>("normal.speed_scale_ratio", 1.0);
    this->declare_parameter<double>("advance.straight_steer_threshold", 0.1);
    this->declare_parameter<double>("advance.straight_speed_scale_ratio", 1.0);
    this->declare_parameter<double>("advance.cornering_speed_scale_ratio", 0.5);
    this->declare_parameter<double>("velocity_threshold_kmh", 34.5);
    this->declare_parameter<double>("throttle_on_limit", 0.0);
    this->declare_parameter<std::vector<double>>("section_steer_scale_ratios", {});

    // パラメータの初期値を取得
    this->get_parameter("filter_type", filter_type_);
    this->get_parameter("window_size", window_size_);
    this->get_parameter("use_scale_filter", use_scale_filter_);
    this->get_parameter("scale_filter_type", scale_filter_type_);
    this->get_parameter("normal.speed_scale_ratio", normal_speed_scale_ratio_);
    this->get_parameter("advance.straight_steer_threshold", advance_straight_steer_threshold_);
    this->get_parameter("advance.straight_speed_scale_ratio", advance_straight_speed_scale_ratio_);
    this->get_parameter("advance.cornering_speed_scale_ratio", advance_cornering_speed_scale_ratio_);
    this->get_parameter("velocity_threshold_kmh", velocity_threshold_kmh_);
    this->get_parameter("throttle_on_limit", throttle_on_limit_);
    this->get_parameter("section_steer_scale_ratios", section_steer_scale_ratios_);


    print_parameters();

    parameters_callback_handle_ = this->add_on_set_parameters_callback(
      std::bind(&AckermannFilterNode::parameters_callback, this, std::placeholders::_1));

    // PublisherとSubscriberの初期化
    publisher_ = this->create_publisher<AckermannMsg>(OUTPUT_TOPIC, 10);
    subscription_ = this->create_subscription<AckermannMsg>(
      INPUT_TOPIC, 10, std::bind(&AckermannFilterNode::topic_callback, this, std::placeholders::_1));
    velocity_subscription_ = this->create_subscription<VelocityMsg>(
      VELOCITY_TOPIC, 10, std::bind(&AckermannFilterNode::velocity_callback, this, std::placeholders::_1));
    status_subscriber_ = this->create_subscription<StatusMsg>(
        STATUS_TOPIC, 10, std::bind(&AckermannFilterNode::status_callback, this, std::placeholders::_1));
  }

private:
  void topic_callback(const AckermannMsg::SharedPtr msg)
  {
    speed_buffer_.push_back(msg->longitudinal.acceleration);
    steering_angle_buffer_.push_back(msg->lateral.steering_tire_angle);

    while (speed_buffer_.size() > static_cast<size_t>(window_size_)) {
      speed_buffer_.pop_front();
      steering_angle_buffer_.pop_front();
    }
    
    auto filtered_msg = *msg;
    
    // 平滑化フィルターの適用
    if (filter_type_ == "average") {
      apply_average_filter(filtered_msg);
    } else if (filter_type_ == "median") {
      apply_median_filter(filtered_msg);
    } else if (!speed_buffer_.empty()) {
      filtered_msg.longitudinal.acceleration = speed_buffer_.back();
      filtered_msg.lateral.steering_tire_angle = steering_angle_buffer_.back();
    }
    
    // アクセルのスケールフィルター適用 
    if (use_scale_filter_) {
      if (scale_filter_type_ == "advance") {
        apply_advanced_scale_filter(filtered_msg);
      } else {
        apply_normal_scale_filter(filtered_msg);
      }
    }

    // セクションに応じたステアリングスケールの適用
    apply_section_steer_scale(filtered_msg);

    // 速度制限ロジック
    if ((current_velocity_mps_ * 3.6) > velocity_threshold_kmh_) {
      filtered_msg.longitudinal.acceleration = throttle_on_limit_;
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 5000,
        "Velocity limit exceeded (%.2f km/h > %.2f km/h)! Forcing acceleration to %.2f.",
        current_velocity_mps_ * 3.6, velocity_threshold_kmh_, throttle_on_limit_);
    }

    publisher_->publish(filtered_msg);
  }

  void velocity_callback(const VelocityMsg::SharedPtr msg)
  {
    current_velocity_mps_ = msg->longitudinal_velocity;
  }
  
  void status_callback(const StatusMsg::SharedPtr msg)
  {
    if (msg->data.size() > 3)
    {
      current_section_ = msg->data[3];
    }
  }
  
  rcl_interfaces::msg::SetParametersResult parameters_callback(
    const std::vector<rclcpp::Parameter> &parameters)
  {
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;
    result.reason = "success";

    for (const auto &param : parameters) {
      const std::string param_name = param.get_name();
      if (param_name == "filter_type") {
        filter_type_ = param.as_string();
      } else if (param_name == "window_size") {
        window_size_ = param.as_int();
      } else if (param_name == "use_scale_filter") {
        use_scale_filter_ = param.as_bool();
      } else if (param_name == "scale_filter_type") {
        scale_filter_type_ = param.as_string();
      } else if (param_name == "normal.speed_scale_ratio") {
        normal_speed_scale_ratio_ = param.as_double();
      } 
      else if (param_name == "advance.straight_steer_threshold") {
        advance_straight_steer_threshold_ = param.as_double();
      } else if (param_name == "advance.straight_speed_scale_ratio") {
        advance_straight_speed_scale_ratio_ = param.as_double();
      } else if (param_name == "advance.cornering_speed_scale_ratio") {
        advance_cornering_speed_scale_ratio_ = param.as_double();
      } 
      else if (param_name == "velocity_threshold_kmh") {
        velocity_threshold_kmh_ = param.as_double();
      } else if (param_name == "throttle_on_limit") {
        throttle_on_limit_ = param.as_double();
      } else if (param_name == "section_steer_scale_ratios") { // ★ 追加
        section_steer_scale_ratios_ = param.as_double_array();
      }
    }
    
    if (result.successful) {
      RCLCPP_INFO(this->get_logger(), "New parameters have been applied.");
      print_parameters();
    }
    return result;
  }

  void print_parameters() {
    RCLCPP_INFO(this->get_logger(), "--- Ackermann Filter Node Settings ---");
    RCLCPP_INFO(this->get_logger(), "Filter type: %s", filter_type_.c_str());
    if (filter_type_ != "none") {
      RCLCPP_INFO(this->get_logger(), "Window size: %d", window_size_);
    }
    RCLCPP_INFO(this->get_logger(), "Use scale filter (for Accel): %s", use_scale_filter_ ? "true" : "false");

    if (use_scale_filter_){
      RCLCPP_INFO(this->get_logger(), "Scale filter type: %s", scale_filter_type_.c_str());
      if (scale_filter_type_ == "advance") {
          RCLCPP_INFO(this->get_logger(), "  [advance] Straight steer threshold: %.2f rad", advance_straight_steer_threshold_);
          RCLCPP_INFO(this->get_logger(), "  [advance] Straight speed scale ratio: %.2f", advance_straight_speed_scale_ratio_);
          RCLCPP_INFO(this->get_logger(), "  [advance] Cornering speed scale ratio: %.2f", advance_cornering_speed_scale_ratio_);
      } else {
          RCLCPP_INFO(this->get_logger(), "  [normal] Speed scale ratio: %.2f", normal_speed_scale_ratio_);
      }
    }
    RCLCPP_INFO(this->get_logger(), "--- Velocity Limiter Settings ---");
    RCLCPP_INFO(this->get_logger(), "Velocity Threshold: %.2f km/h", velocity_threshold_kmh_);
    RCLCPP_INFO(this->get_logger(), "Throttle on Limit: %.2f", throttle_on_limit_);
    
    RCLCPP_INFO(this->get_logger(), "--- Section Steer Scale Settings ---");
    std::string scales_str = "[";
    for(size_t i = 0; i < section_steer_scale_ratios_.size(); ++i) {
        scales_str += "Section " + std::to_string(i) + ": " + std::to_string(section_steer_scale_ratios_[i]);
        if (i < section_steer_scale_ratios_.size() - 1) scales_str += ", ";
    }
    scales_str += "]";
    RCLCPP_INFO(this->get_logger(), "Ratios: %s", scales_str.c_str());

    RCLCPP_INFO(this->get_logger(), "------------------------------------");
  }

  // ステアリングのスケール処理を削除
  void apply_normal_scale_filter(AckermannMsg &msg)
  {
    msg.longitudinal.acceleration *= normal_speed_scale_ratio_;
  }

  // ステアリングのスケール処理を削除
  void apply_advanced_scale_filter(AckermannMsg &msg)
  {
    if (std::fabs(msg.lateral.steering_tire_angle) < advance_straight_steer_threshold_) {
      msg.longitudinal.acceleration *= advance_straight_speed_scale_ratio_;
    } else {
      msg.longitudinal.acceleration *= advance_cornering_speed_scale_ratio_;
    }
  }

  // セクションに応じてステアリングをスケールする関数
  void apply_section_steer_scale(AckermannMsg &msg)
  {
    // current_section_を整数に変換してインデックスとして使用
    int section_index = static_cast<int>(current_section_);

    // インデックスがパラメータ配列の範囲内かチェック
    if (section_index >= 0 && static_cast<size_t>(section_index) < section_steer_scale_ratios_.size())
    {
      // 対応するスケール値を適用
      msg.lateral.steering_tire_angle *= section_steer_scale_ratios_[section_index];
    }
  }


  void apply_average_filter(AckermannMsg &msg)
  {
    if (speed_buffer_.empty()) return;
    double speed_sum = std::accumulate(speed_buffer_.begin(), speed_buffer_.end(), 0.0);
    double steer_sum = std::accumulate(steering_angle_buffer_.begin(), steering_angle_buffer_.end(), 0.0);
    msg.longitudinal.acceleration = speed_sum / speed_buffer_.size();
    msg.lateral.steering_tire_angle = steer_sum / steering_angle_buffer_.size();
  }

  void apply_median_filter(AckermannMsg &msg)
  {
    if (speed_buffer_.empty()) return;
    msg.longitudinal.acceleration = calculate_median(speed_buffer_);
    msg.lateral.steering_tire_angle = calculate_median(steering_angle_buffer_);
  }
  
  double calculate_median(const std::deque<double>& data)
  {
    if (data.empty()) return 0.0;
    std::vector<double> sorted_data(data.begin(), data.end());
    size_t n = sorted_data.size();
    std::sort(sorted_data.begin(), sorted_data.end());
    if (n % 2 == 0) {
      return (sorted_data[n / 2 - 1] + sorted_data[n / 2]) / 2.0;
    } else {
      return sorted_data[n / 2];
    }
  }


  rclcpp::Subscription<AckermannMsg>::SharedPtr subscription_;
  rclcpp::Publisher<AckermannMsg>::SharedPtr publisher_;
  rclcpp::Subscription<VelocityMsg>::SharedPtr velocity_subscription_;
  rclcpp::Subscription<StatusMsg>::SharedPtr status_subscriber_;
  OnSetParametersCallbackHandle::SharedPtr parameters_callback_handle_;
  
  // メンバ変数
  std::string filter_type_;
  int window_size_;
  bool use_scale_filter_;
  std::string scale_filter_type_;
  double normal_speed_scale_ratio_;
  double advance_straight_steer_threshold_;
  double advance_straight_speed_scale_ratio_;
  double advance_cornering_speed_scale_ratio_;
  double velocity_threshold_kmh_;
  double throttle_on_limit_;
  double current_velocity_mps_;
  float current_section_;

  std::vector<double> section_steer_scale_ratios_;


  std::deque<double> speed_buffer_;
  std::deque<double> steering_angle_buffer_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<AckermannFilterNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}