cmake_minimum_required(VERSION 2.8)
project(hrnet_segment)

# list catkin dependencies
set(_ROS_PKG_DEPS
  sensor_msgs
  rospy)


# Find catkin macros and libraries, also other catkin packages.
find_package(catkin REQUIRED COMPONENTS ${_ROS_PKG_DEPS})
include_directories(include ${catkin_INCLUDE_DIRS})
catkin_python_setup()

catkin_package(CATKIN_DEPENDS ${_ROS_PKG_DEPS})

install(DIRECTORY src/tools/
        DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
        USE_SOURCE_PERMISSIONS)

install(DIRECTORY launch/
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch)

# unit tests
if (CATKIN_ENABLE_TESTING)
  # check all the launch files
  find_package(roslaunch REQUIRED)
  roslaunch_add_file_check(launch)

endif (CATKIN_ENABLE_TESTING)
