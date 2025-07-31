#include <iostream>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32.hpp>
#include <autoware_auto_control_msgs/msg/ackermann_control_command.hpp>

class ControlCommandPublisher : public rclcpp::Node
{
public:
    ControlCommandPublisher()
        : Node("control_command_publisher"),
          handle_position_(0.0f),
          throttle_position_(0.0f),
          brake_position_(0.0f)
    {
        this->declare_parameter<float>("accel_scale", 1.0);
        this->declare_parameter<float>("brake_scale", 1.0);

        this->get_parameter("accel_scale", accel_scale_);
        this->get_parameter("brake_scale", brake_scale_);
        
        RCLCPP_INFO(this->get_logger(), "Parameter accel_scale set to: %f", accel_scale_);
        RCLCPP_INFO(this->get_logger(), "Parameter brake_scale set to: %f", brake_scale_);
        
        // トピックのサブスクライバ
        handle_subscriber_ = this->create_subscription<std_msgs::msg::Float32>(
            "handle_position", 1,
            std::bind(&ControlCommandPublisher::handle_callback, this, std::placeholders::_1));
        
        throttle_subscriber_ = this->create_subscription<std_msgs::msg::Float32>(
            "throttle_position", 1,
            std::bind(&ControlCommandPublisher::throttle_callback, this, std::placeholders::_1));
        
        brake_subscriber_ = this->create_subscription<std_msgs::msg::Float32>(
            "brake_position", 1,
            std::bind(&ControlCommandPublisher::brake_callback, this, std::placeholders::_1));
        
        // AckermannControlCommandのパブリッシャー
        control_command_publisher_ = this->create_publisher<autoware_auto_control_msgs::msg::AckermannControlCommand>(
            "/awsim/control_cmd", 1);
        
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(10),
            std::bind(&ControlCommandPublisher::publish_control_command, this));
    }

private:
    void handle_callback(const std_msgs::msg::Float32::SharedPtr msg)
    {
        handle_position_ = msg->data;
    }

    void throttle_callback(const std_msgs::msg::Float32::SharedPtr msg)
    {
        throttle_position_ = msg->data;
    }

    void brake_callback(const std_msgs::msg::Float32::SharedPtr msg)
    {
        brake_position_ = msg->data;
    }

    void publish_control_command()
    {
        auto command_msg = autoware_auto_control_msgs::msg::AckermannControlCommand();
        command_msg.stamp = this->get_clock()->now();
        command_msg.longitudinal.stamp = command_msg.stamp;
        command_msg.lateral.stamp = command_msg.stamp;

        command_msg.lateral.steering_tire_angle = -handle_position_;
        command_msg.longitudinal.acceleration = accel_scale_ * throttle_position_ - brake_scale_ * brake_position_;

        control_command_publisher_->publish(command_msg);
    }

    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr handle_subscriber_;
    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr throttle_subscriber_;
    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr brake_subscriber_;
    rclcpp::Publisher<autoware_auto_control_msgs::msg::AckermannControlCommand>::SharedPtr control_command_publisher_;
    rclcpp::TimerBase::SharedPtr timer_;

    // パラメータを格納するメンバ変数
    float accel_scale_;
    float brake_scale_;

    float handle_position_;
    float throttle_position_;
    float brake_position_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ControlCommandPublisher>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}