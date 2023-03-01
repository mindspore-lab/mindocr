#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
# Description: mxOCR c++ build.
# Author: MindX SDK
# Create: 2022
# History: NA

set -e

current_path=$(cd $(dirname $0); pwd)
build_type="Release"

os_type=$(arch)
if [ "${os_type}" = 'aarch64' ];then
	export ARCH_PATTERN=aarch64-linux
else
	export ARCH_PATTERN=x86_64-linux
fi

function main()
{
  local build_path=${current_path}/build
  [ -d "${build_path}" ] && rm -rf ${build_path}
  mkdir -p ${build_path}
  cd ${build_path}

  cmake -DCMAKE_BUILD_TYPE=$build_type ..
  make -j
}

main
if [ $? -ne 0 ]; then
  echo "Build failed."
  exit 1
fi

exit 0