cmake_minimum_required(VERSION 3.8)
project(scan_node)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

add_executable(scan_node src/scan_node.cpp)
ament_target_dependencies(scan_node rclcpp sensor_msgs nav_msgs geometry_msgs tf2_geometry_msgs visualization_msgs)
install(TARGETS scan_node
  DESTINATION lib/${PROJECT_NAME}
)


install(DIRECTORY
  launch
  config
  map
  DESTINATION share/${PROJECT_NAME})  

ament_package()

