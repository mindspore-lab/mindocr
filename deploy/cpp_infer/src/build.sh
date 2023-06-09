#!/bin/bash
set -e

current_path=$(cd $(dirname $0); pwd)
build_type="Release"

os_type=$(arch)
if [ "${os_type}" = 'aarch64' ];then
	export ARCH_PATTERN=aarch64-linux
else
	export ARCH_PATTERN=x86_64-linux
fi

function downloadClipper() {
    if [ -e ./data_process/postprocess/clipper.cpp ]; then
        echo "clipper exists"
    else
      mkdir -p clipper
      wget https://udomain.dl.sourceforge.net/project/polyclipping/clipper_ver6.4.2.zip --no-check-certificate

      unzip clipper_ver6.4.2.zip -d ./clipper
      cp ./clipper/cpp/clipper.cpp ./data_process/postprocess
      cp ./clipper/cpp/clipper.hpp ./data_process/postprocess

      rm clipper_ver6.4.2.zip
      rm -rf clipper
    fi
}

function main()
{
  local build_path=${current_path}/build
  [ -d "${build_path}" ] && rm -rf ${build_path}
  mkdir -p ${build_path}
  cd ${build_path}

  cmake -DCMAKE_BUILD_TYPE=$build_type ..
  make -j
}
downloadClipper
main
if [ $? -ne 0 ]; then
  echo "Build failed."
  exit 1
fi

exit 0
