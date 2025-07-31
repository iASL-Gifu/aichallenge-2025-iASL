#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joy.hpp"
#include "std_msgs/msg/bool.hpp"

class JoyRecorderTriggerNode : public rclcpp::Node
{
public:
  JoyRecorderTriggerNode()
  : Node("joy_recorder_trigger_node"),
    prev_start_pressed_(false),
    prev_stop_pressed_(false)
  {
    // --- パラメータ宣言＆取得 ---
    // rosbag2_recorderの記録開始/停止に使用するボタンのインデックス
    declare_parameter<int>("start_button_index", 9);
    declare_parameter<int>("stop_button_index",  8);

    get_parameter("start_button_index", start_button_index_);
    get_parameter("stop_button_index", stop_button_index_);

    RCLCPP_INFO(this->get_logger(), "Using START button index: %d", start_button_index_);
    RCLCPP_INFO(this->get_logger(), "Using STOP button index: %d", stop_button_index_);

    // --- サブスクライバ／パブリッシャ設定 ---
    joy_sub_ = create_subscription<sensor_msgs::msg::Joy>(
      "/joy", 10, std::bind(&JoyRecorderTriggerNode::joy_callback, this, std::placeholders::_1));

    trigger_pub_ = create_publisher<std_msgs::msg::Bool>("/rosbag2_recorder/trigger", 10);
  }

private:
  /**
   * @brief ボタンが押された瞬間だけtrueを返すヘルパー関数（デバウンス処理）
   * @param curr 現在のボタン状態 (true: 押されている, false: 押されていない)
   * @param prev_flag 前回のボタン状態を保持するフラグへの参照
   * @return ボタンが押された瞬間であればtrue
   */
  bool check_button_press(bool curr, bool &prev_flag)
  {
    if (curr && !prev_flag) {
      prev_flag = true;
      return true;
    } else if (!curr) {
      prev_flag = false;
    }
    return false;
  }

  /**
   * @brief /joyトピックのコールバック関数
   * @param msg 受信したJoyメッセージ
   */
  void joy_callback(const sensor_msgs::msg::Joy::SharedPtr msg)
  {
    // ボタンのインデックスがメッセージの範囲内か確認
    if (msg->buttons.size() <= start_button_index_ || msg->buttons.size() <= stop_button_index_) {
      RCLCPP_WARN_ONCE(this->get_logger(), "Button index is out of range. Please check your joy device and parameters.");
      return;
    }

    // start/stop ボタンの状態を取得（デバウンス付き）
    bool curr_start = (msg->buttons[start_button_index_] == 1);
    bool curr_stop  = (msg->buttons[stop_button_index_]  == 1);

    // STARTボタンが押されたら、rosbag記録開始トリガーを送信
    if (check_button_press(curr_start, prev_start_pressed_)) {
      std_msgs::msg::Bool trigger_msg;
      trigger_msg.data = true;
      trigger_pub_->publish(trigger_msg);
      RCLCPP_INFO(this->get_logger(), "Published rosbag2 trigger: START recording");
    }

    // STOPボタンが押されたら、rosbag記録停止トリガーを送信
    if (check_button_press(curr_stop, prev_stop_pressed_)) {
      std_msgs::msg::Bool trigger_msg;
      trigger_msg.data = false;
      trigger_pub_->publish(trigger_msg);
      RCLCPP_INFO(this->get_logger(), "Published rosbag2 trigger: STOP recording");
    }
  }

  // --- メンバ変数 ---
  // サブスクライバ／パブリッシャ
  rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joy_sub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr      trigger_pub_;

  // パラメータ
  int start_button_index_;
  int stop_button_index_;

  // ボタンの連射防止用フラグ
  bool prev_start_pressed_;
  bool prev_stop_pressed_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<JoyRecorderTriggerNode>());
  rclcpp::shutdown();
  return 0;
}