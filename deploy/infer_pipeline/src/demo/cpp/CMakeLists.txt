# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
cmake_minimum_required(VERSION 3.10.2)
project(OcrDemo)

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
add_definitions(-Dgoogle=mindxsdk_private)

set(PROJECT_SRC_ROOT ${CMAKE_CURRENT_LIST_DIR})
set(CMAKE_MODULE_PATH ${PROJECT_SRC_ROOT}/CMake)
set(CMAKE_SKIP_BUILD_RPATH True)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SRC_ROOT}/dist)
add_compile_options(-std=c++11 -fPIE -fstack-protector-all -Wall)
add_definitions(-DENABLE_DVPP_INTERFACE)

set(AscendBaseFolder ${PROJECT_SRC_ROOT}/ascendbase)

# Find ascendbase
set(ASCEND_BASE_DIR ${AscendBaseFolder})
get_filename_component(ASCEND_BASE_ABS_DIR ${ASCEND_BASE_DIR} ABSOLUTE)

file(GLOB_RECURSE ASCEND_BASE_SRC_FILES
		${ASCEND_BASE_ABS_DIR}/BlockingQueue/*cpp
		${ASCEND_BASE_ABS_DIR}/ConfigParser/*cpp
		${ASCEND_BASE_ABS_DIR}/Framework/ModuleManager/*cpp
		${ASCEND_BASE_ABS_DIR}/CommandParser/*cpp
		)

# Find Header
set(HEADER_FILE_DIR
		${ASCEND_BASE_ABS_DIR}
		${ASCEND_BASE_ABS_DIR}/Framework
		${PROJECT_SRC_ROOT}/Modules
		${PROJECT_SRC_ROOT}/Modules/CrnnPost
		${PROJECT_SRC_ROOT}/Modules/DbnetPost
		${PROJECT_SRC_ROOT}/Modules/Processors
		${PROJECT_SRC_ROOT}/Modules/Signal
		${PROJECT_SRC_ROOT}/Modules/Utils
		)
include_directories(${HEADER_FILE_DIR})

# Find Src
file(GLOB_RECURSE SRC_FILE
		${PROJECT_SRC_ROOT}/main.cpp
		${PROJECT_SRC_ROOT}/Modules/Processors/HandOutProcess/*.cpp
		${PROJECT_SRC_ROOT}/Modules/Processors/DbnetPreProcess/*.cpp
		${PROJECT_SRC_ROOT}/Modules/Processors/DbnetInferProcess/*.cpp
		${PROJECT_SRC_ROOT}/Modules/Processors/DbnetPostProcess/*.cpp
		${PROJECT_SRC_ROOT}/Modules/Processors/ClsPreProcess/*.cpp
		${PROJECT_SRC_ROOT}/Modules/Processors/ClsInferProcess/*.cpp
		${PROJECT_SRC_ROOT}/Modules/Processors/ClsPostProcess/*.cpp
		${PROJECT_SRC_ROOT}/Modules/Processors/CrnnPreProcess/*.cpp
		${PROJECT_SRC_ROOT}/Modules/Processors/CrnnInferProcess/*.cpp
		${PROJECT_SRC_ROOT}/Modules/Processors/CrnnPostProcess/*.cpp
		${PROJECT_SRC_ROOT}/Modules/Processors/CollectProcess/*.cpp
		${PROJECT_SRC_ROOT}/Modules/CrnnPost/*.cpp
		${PROJECT_SRC_ROOT}/Modules/DbnetPost/*.cpp
		${PROJECT_SRC_ROOT}/Modules/Signal/*.cpp
		${PROJECT_SRC_ROOT}/Modules/Utils/*.cpp
		)

set(SOURCE_FILE
		${SRC_FILE}
		${ASCEND_BASE_SRC_FILES}
		)

set(MX_SDK_HOME $ENV{MX_SDK_HOME})
set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
set(ARCH_PATTERN $ENV{ARCH_PATTERN})
include_directories(
		${MX_SDK_HOME}/include
		${MX_SDK_HOME}/opensource/include
		${MX_SDK_HOME}/opensource/include/opencv4
		${MX_SDK_HOME}/include/MxBase
		${ASCEND_HOME_PATH}/${ARCH_PATTERN}/include
		${ASCEND_HOME_PATH}/${ARCH_PATTERN}/runtime/include
)

link_directories(
		${MX_SDK_HOME}/lib
		${MX_SDK_HOME}/lib/modelpostprocessors
		${MX_SDK_HOME}/include
		${MX_SDK_HOME}/opensource/lib
		${ASCEND_HOME_PATH}/atc/lib64
)

# Set the target executable file
add_executable(main ${SOURCE_FILE})

target_link_libraries(main ascendcl acl_dvpp opencv_world mxbase cpprest glog pthread -Wl,-z,relro,-z,now,-z,noexecstack -pie -s)
