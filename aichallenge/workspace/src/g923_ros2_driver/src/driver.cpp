#include <iostream>
#include <string>
#include <vector>
#include <map> 
#include <fcntl.h>
#include <unistd.h>
#include <linux/input.h>
#include <linux/input-event-codes.h> 

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/int32.hpp>
#include <sensor_msgs/msg/joy.hpp> 

class HandleInputPublisher : public rclcpp::Node
{
public:
    HandleInputPublisher(const std::string &device_path)
        : Node("handle_input_publisher"), device_path_(device_path)
    {
        // 既存のPublisher
        handle_publisher_ = this->create_publisher<std_msgs::msg::Float32>("handle_position", 10);
        throttle_publisher_ = this->create_publisher<std_msgs::msg::Float32>("throttle_position", 10);
        brake_publisher_ = this->create_publisher<std_msgs::msg::Float32>("brake_position", 10);
        gear_publisher_ = this->create_publisher<std_msgs::msg::Int32>("gear_position", 10);
        joy_publisher_ = this->create_publisher<sensor_msgs::msg::Joy>("joy", 10);

        // 状態変数の初期化
        handle_position_ = 0.0f;
        throttle_position_ = 0.0f;
        brake_position_ = 0.0f;
        gear_position_ = 0; 

        initialize_button_map();
        joy_state_.buttons.resize(11, 0);
        joy_state_.axes.resize(2, 0.0f);

        open_device();
        
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(10),
            std::bind(&HandleInputPublisher::publish_latest_state, this)
        );
    }

    void open_device()
    {
        fd_ = open(device_path_.c_str(), O_RDONLY | O_NONBLOCK); // ノンブロッキングに変更
        if (fd_ == -1)
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to open device: %s", device_path_.c_str());
            rclcpp::shutdown();
        }
    }

    void read_loop()
    {
        struct input_event ev;
        // read()がブロックしないので、ループ周期を制御できる
        while (rclcpp::ok())
        {
            if (read(fd_, &ev, sizeof(ev)) == sizeof(ev))
            {
                if (ev.type == EV_ABS)
                {
                    process_abs_event(ev); // ABSイベント処理を関数に分離
                }
                else if (ev.type == EV_KEY)
                {
                    process_key_event(ev); // KEYイベント処理を関数に分離
                }
            }
            // CPU使用率が高くなりすぎないように少し待つ
            std::this_thread::sleep_for(std::chrono::microseconds(500));
        }
    }

private:
    std::string device_path_;
    int fd_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr handle_publisher_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr throttle_publisher_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr brake_publisher_;
    rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr gear_publisher_;
    rclcpp::TimerBase::SharedPtr timer_;

    rclcpp::Publisher<sensor_msgs::msg::Joy>::SharedPtr joy_publisher_;
    sensor_msgs::msg::Joy joy_state_;
    std::map<int, int> button_map_;

    float handle_position_;
    float throttle_position_;
    float brake_position_;
    int gear_position_;

    void initialize_button_map()
    {
        // evtestで確認したボタンコードとJoyメッセージのインデックスを対応付ける
        button_map_ = {
            {BTN_SOUTH,  0}, // A (Cross on PS)
            {BTN_EAST,   1}, // B (Circle on PS)
            {BTN_WEST,   2}, // X (Square on PS)
            {BTN_NORTH,  3}, // Y (Triangle on PS)
            {BTN_TL,     4}, // L1
            {BTN_TR,     5}, // R1
            {BTN_SELECT, 6}, // Share
            {BTN_START,  7}, // Options
            {BTN_MODE,   8}, // PS Button
            {BTN_THUMBL, 9}, // L3
            {BTN_THUMBR, 10} // R3
        };
    }

    void process_abs_event(const struct input_event& ev)
    {
        switch (ev.code) {
            case ABS_X:
                handle_position_ = normalize_handle(ev.value);
                break;
            case ABS_Z:
                throttle_position_ = normalize_pedal(ev.value);
                break;
            case ABS_Y:
                brake_position_ = normalize_pedal(ev.value);
                break;
            
            case ABS_HAT0X:
                joy_state_.axes[0] = static_cast<float>(ev.value);
                break;
            case ABS_HAT0Y:
                joy_state_.axes[1] = static_cast<float>(ev.value) * -1; 
                break;
        }
    }

    void process_key_event(const struct input_event& ev)
    {
        // ギアの判定
        int new_gear = determine_gear_position(ev.code, ev.value);
        if (new_gear != -1) { 
            gear_position_ = new_gear;
            return; 
        }

        // その他のボタンの判定
        auto it = button_map_.find(ev.code);
        if (it != button_map_.end()) {
            int button_index = it->second;
            // ev.value は 1 (press), 0 (release)
            joy_state_.buttons[button_index] = (ev.value > 0) ? 1 : 0;
        }
    }
    // ▲▲▲ 追加ここまで ▲▲▲


    float normalize_handle(int value, int max_value = 65535)
    {
        return (static_cast<float>(value) / max_value) * 2.0f - 1.0f;
    }

    float normalize_pedal(int value, int max_value = 255)
    {
        return 1.0f - (static_cast<float>(value) / max_value);
    }

   
    int determine_gear_position(int code, int value)
    {
        if (value == 1) // キーが押された瞬間にのみ判定
        {
            switch (code)
            {
                case BTN_TRIGGER_HAPPY1: return 1; 
                case BTN_TRIGGER_HAPPY2: return 2;
                case BTN_TRIGGER_HAPPY3: return 3;
                case BTN_TRIGGER_HAPPY4: return 4;
                case BTN_TRIGGER_HAPPY5: return 5;
                case BTN_TRIGGER_HAPPY6: return 6;
                default: return -1; // ギア関連のボタンではない
            }
        }
        return -1; 
    }

    void publish_latest_state()
    {
        publish_float(handle_publisher_, handle_position_);
        publish_float(throttle_publisher_, throttle_position_);
        publish_float(brake_publisher_, brake_position_);
        publish_int(gear_publisher_, gear_position_);
        
        joy_state_.header.stamp = this->get_clock()->now();
        joy_publisher_->publish(joy_state_);
    }

    void publish_float(rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr publisher, float value)
    {
        auto message = std_msgs::msg::Float32();
        message.data = value;
        publisher->publish(message);
    }

    void publish_int(rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr publisher, int value)
    {
        auto message = std_msgs::msg::Int32();
        message.data = value;
        publisher->publish(message);
    }
};

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <device_path>" << std::endl;
        return 1;
    }

    std::string device_path = argv[1];

    rclcpp::init(argc, argv);
    auto node = std::make_shared<HandleInputPublisher>(device_path);

    // read_loopを別スレッドで実行
    std::thread read_thread([&node]() { node->read_loop(); });
    
    rclcpp::spin(node);

    read_thread.join(); 
    rclcpp::shutdown();

    return 0;
}