# G923_Converter
Aichallenge-2024のAWSIMにG923を対応させるためのパッケージです。ubuntu22.04での動作を確認しています。  

## 事前準備
1. g923_ros2_driver
```shell
#ワークスペースの作成(まだ作ってない場合)
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src

#レポジトリのクローン
git clone https://github.com/Arata-Stu/g923_ros2_driver.git

#ビルド
cd ~/ros2_ws/
colcon build --packages-select g923_ros2_driver
```
2. g923のinputパスの確認
```shell
#evtestをインストール
sudo apt-get install evtest
#実行
evtest
#G923を探し、数字を確認(後に実行時に利用する)
#
```
## パッケージのビルド、実行
```shell
#ワークスペースの作成(まだ作ってない場合)
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src

#レポジトリのクローン
git clone https://github.com/Arata-Stu/g923_converter.git

#ビルド
cd ~/ros2_ws/
colcon build --packages-select g923_converter

#実行
source install/setup.bash

# eventXを変更　例 /dev/input/event1
ros2 launch converter.launch.xml device_path:=/dev/input/eventX

```