#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/int32.hpp>
#include <std_msgs/msg/bool.hpp>

class GearPositionSubscriber : public rclcpp::Node
{
public:
    GearPositionSubscriber()
        : Node("gear_changer_node")
    {
        subscription_ = this->create_subscription<std_msgs::msg::Int32>(
            "gear_position", 1,
            std::bind(&GearPositionSubscriber::topic_callback, this, std::placeholders::_1));

        control_publisher_ = this->create_publisher<std_msgs::msg::Bool>("/control/control_mode_request_topic", 10);
    }

private:
    void topic_callback(const std_msgs::msg::Int32::SharedPtr msg) const
    {
        auto message = std_msgs::msg::Bool();
        message.data = (msg->data % 2 != 0);  // 奇数ならTrue、偶数ならFalse
        control_publisher_->publish(message);

        RCLCPP_INFO(this->get_logger(), "Current gear position: %d, Control mode request: %s", msg->data, message.data ? "True" : "False");
    }

    rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr subscription_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr control_publisher_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<GearPositionSubscriber>());
    rclcpp::shutdown();
    return 0;
}
