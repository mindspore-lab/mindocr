cmake_minimum_required(VERSION 3.10.2)
project(mindocr_cpp_infer)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
add_definitions(-Dgoogle=mindxsdk_private)

set(PROJECT_SRC_ROOT ${CMAKE_CURRENT_LIST_DIR})
set(CMAKE_MODULE_PATH ${PROJECT_SRC_ROOT}/CMake)
set(CMAKE_SKIP_BUILD_RPATH True)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SRC_ROOT}/dist)
add_compile_options(-std=c++14 -fPIE -fstack-protector-all -Wall)
add_definitions(-DENABLE_DVPP_INTERFACE)

# Find Header
set(HEADER_FILE_DIR
		${PROJECT_SRC_ROOT}
		${PROJECT_SRC_ROOT}/base
		${PROJECT_SRC_ROOT}/data_process
		${PROJECT_SRC_ROOT}/data_process/postprocess
		${PROJECT_SRC_ROOT}/parallel
		${PROJECT_SRC_ROOT}/parallel/module
		${PROJECT_SRC_ROOT}/profile
		${PROJECT_SRC_ROOT}/utils
		)
include_directories(${HEADER_FILE_DIR})

# Find Src
file(GLOB_RECURSE SRC_FILE
		${PROJECT_SRC_ROOT}/base/command_parser/*.cpp
		${PROJECT_SRC_ROOT}/base/config_parser/*.cpp
		${PROJECT_SRC_ROOT}/base/config_parser/*.cpp
		${PROJECT_SRC_ROOT}/data_process/postprocess/*.cpp
		${PROJECT_SRC_ROOT}/parallel/framework/*.cpp
		${PROJECT_SRC_ROOT}/parallel/module/classification/*.cpp
		${PROJECT_SRC_ROOT}/parallel/module/common/*.cpp
		${PROJECT_SRC_ROOT}/parallel/module/detection/*.cpp
		${PROJECT_SRC_ROOT}/parallel/module/recognition/*.cpp
		${PROJECT_SRC_ROOT}/profile/*.cpp
		${PROJECT_SRC_ROOT}/utils/*.cpp
		${PROJECT_SRC_ROOT}/main.cpp
		)

set(SOURCE_FILE
		${SRC_FILE})

set(MX_SDK_HOME $ENV{MX_SDK_HOME})
set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
set(ARCH_PATTERN $ENV{ARCH_PATTERN})
set(LITE_HOME $ENV{LITE_HOME})
include_directories(
		${MX_SDK_HOME}/include
		${MX_SDK_HOME}/opensource/include
		${MX_SDK_HOME}/opensource/include/opencv4
		${MX_SDK_HOME}/include/MxBase
		${ASCEND_HOME_PATH}/${ARCH_PATTERN}/include
		${ASCEND_HOME_PATH}/${ARCH_PATTERN}/runtime/include
		${LITE_HOME}/runtime/
)

link_directories(
		${MX_SDK_HOME}/lib
		${MX_SDK_HOME}/lib/modelpostprocessors
		${MX_SDK_HOME}/include
		${MX_SDK_HOME}/opensource/lib
		${ASCEND_HOME_PATH}/atc/lib64
		${LITE_HOME}/runtime/lib
		${LITE_HOME}/tools/converter/lib
)
include_directories(/usr/local/Ascend/latest/runtime/include)
# Set the target executable file
add_executable(infer ${SOURCE_FILE})
target_link_libraries(infer  gflags /usr/local/Ascend/latest/runtime/lib64/libascendcl.so)
target_link_libraries(infer mindspore-lite dl ascendcl acl_dvpp opencv_world mxbase cpprest glog pthread securec -Wl,-z,relro,-z,now,-z,noexecstack -pie -s)
