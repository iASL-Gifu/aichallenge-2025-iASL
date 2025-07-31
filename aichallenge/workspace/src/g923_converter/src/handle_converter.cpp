#include <iostream>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32.hpp>
#include <autoware_auto_control_msgs/msg/ackermann_control_command.hpp>

class ControlCommandPublisher : public rclcpp::Node
{
public:
    ControlCommandPublisher()
        : Node("control_command_publisher")
    {
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
            "/control/command/control_cmd", 1);
        
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100),
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
        command_msg.lateral.steering_tire_angle = -handle_position_;
        command_msg.longitudinal.acceleration = alpha * throttle_position_ - beta * brake_position_;  // 仮の計算。必要に応じて修正。

        control_command_publisher_->publish(command_msg);
    }

    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr handle_subscriber_;
    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr throttle_subscriber_;
    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr brake_subscriber_;
    rclcpp::Publisher<autoware_auto_control_msgs::msg::AckermannControlCommand>::SharedPtr control_command_publisher_;
    rclcpp::TimerBase::SharedPtr timer_;

    float alpha = 3.2;
    float beta = 5.0;
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

