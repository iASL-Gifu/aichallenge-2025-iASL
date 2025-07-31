# G923_ROS2_DRIVER
G923をROS2対応させることを目的としたパッケージです。
ubuntu22.04で動作を確認しています。

## topic
| 入力 | topic | 値 
| ---- | ---- | --- 
| ハンドル | /handle_position | float32 -1~1
| アクセルペダル | /throttle_position | float32 0~1
| ブレーキペダル | /brake_position | float32 0~1
| シフトレバー | /gear_position | int32 0~6
## 事前準備
1. oversteerのbuild
```shell

sudo apt install python3 python3-distutils python3-gi python3-gi-cairo python3-pyudev python3-xdg python3-evdev gettext meson appstream-util desktop-file-utils python3-matplotlib python3-scipy
#レポジトリのcloneとbuild
git clone https://github.com/berarma/oversteer.git
cd oversteer
meson setup build
cd build
ninja install

#再起動
sudo reboot
```
2. new-lg4ffのinstall
```shell
sudo apt-get install dkms
sudo git clone https://github.com/berarma/new-lg4ff.git /usr/src/new-lg4ff
sudo dkms install /usr/src/new-lg4ff
sudo update-initramfs -u
```

## パッケージのビルド、実行
```shell
#ワークスペースの作成(まだ作ってない場合)
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src

#レポジトリのクローン
git clone https://github.com/Arata-Stu/g923_ros2_driver.git

#ビルド
cd ~/ros2_ws/
colcon build --packages-select g923_ros2_driver

#実行
source install/setup.bash
ros2 run g923_ros2_driver g923_driver_node
```
