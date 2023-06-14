#!/bin/bash

set -e

BASEPATH=$(cd "$(dirname $0)"; pwd)
OUTPUT_PATH="${BASEPATH}/output"
PYTHON=$(which python3)


mk_new_dir() {
    local create_dir="$1"  # the target to make

    if [[ -d "${create_dir}" ]];then
        rm -rf "${create_dir}"
    fi

    mkdir -pv "${create_dir}"
}

write_checksum() {
    cd "$OUTPUT_PATH" || exit
    PACKAGE_LIST=$(ls mindocr-*.whl) || exit
    for PACKAGE_NAME in $PACKAGE_LIST; do
        echo $PACKAGE_NAME
        sha256sum -b "$PACKAGE_NAME" >"$PACKAGE_NAME.sha256"
    done
}

mk_new_dir "${OUTPUT_PATH}"

${PYTHON} ${BASEPATH}/setup.py bdist_wheel

mv ${BASEPATH}/dist/*whl ${OUTPUT_PATH}

write_checksum


echo "------Successfully created mindocr package------"
