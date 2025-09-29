cmake_minimum_required(VERSION 3.8)
project(data_record_launch)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME})  

ament_package()

