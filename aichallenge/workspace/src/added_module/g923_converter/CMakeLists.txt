cmake_minimum_required(VERSION 3.8)
project(g923_converter)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()


# find dependencies
find_package(ament_cmake_auto REQUIRED)

ament_auto_find_build_dependencies()

ament_auto_add_executable(gear_change_node
  src/gear_changer_node.cpp
)

ament_auto_add_executable(handle_converter_node
src/handle_converter.cpp
)

install(TARGETS
  gear_change_node
  handle_converter_node
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

ament_auto_package(
  INSTALL_TO_SHARE
)