#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/int32.hpp>
#include "autoware_auto_vehicle_msgs/msg/gear_command.hpp" 

class GearCommandController : public rclcpp::Node
{
public:
    GearCommandController()
        : Node("gear_command_controller_node")
    {
        subscription_ = this->create_subscription<std_msgs::msg::Int32>(
            "gear_position", 10, 
            std::bind(&GearCommandController::gear_input_callback, this, std::placeholders::_1));

        gear_command_publisher_ = this->create_publisher<autoware_auto_vehicle_msgs::msg::GearCommand>("/control/command/gear_cmd", 10);
    }

private:
    void gear_input_callback(const std_msgs::msg::Int32::SharedPtr msg)
    {
        // GearCommandメッセージを作成
        auto gear_cmd = autoware_auto_vehicle_msgs::msg::GearCommand();
        gear_cmd.stamp = this->get_clock()->now(); // タイムスタンプを設定

        // 受け取った整数に応じてギア指令を決定
        switch (msg->data)
        {
        case 2:
            gear_cmd.command = autoware_auto_vehicle_msgs::msg::GearCommand::DRIVE;
            break;
        case 3:
            gear_cmd.command = autoware_auto_vehicle_msgs::msg::GearCommand::REVERSE;
            break;
        case 4:
            gear_cmd.command = autoware_auto_vehicle_msgs::msg::GearCommand::PARK;
            break;
        case 1:
            gear_cmd.command = autoware_auto_vehicle_msgs::msg::GearCommand::NEUTRAL;
            break;
        default:
            RCLCPP_WARN(this->get_logger(), "Invalid gear value received: %d. No command sent.", msg->data);
            return; 
        }

        // GearCommandをPublish
        gear_command_publisher_->publish(gear_cmd);

        RCLCPP_INFO(this->get_logger(), "Received gear input: %d -> Publishing GearCommand: %d", msg->data, gear_cmd.command);
    }

    rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr subscription_;
    rclcpp::Publisher<autoware_auto_vehicle_msgs::msg::GearCommand>::SharedPtr gear_command_publisher_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<GearCommandController>());
    rclcpp::shutdown();
    return 0;
}