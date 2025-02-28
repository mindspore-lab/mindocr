#!/bin/bash

EN_PATH="$PWD"/docs/en
ZH_PATH="$PWD"/docs/zh

replace_path(){
  path=$1
  before_arr=($2)

  count=${#before_arr[*]}
  sed -i "s|${before_arr[0]}|${common_after_arr[0]}|g" "$path"/index.md
  sed -i "s|${before_arr[1]}|${common_after_arr[1]}|g" "$path"/mkdocs/customize_dataset.md
  sed -i "s|${before_arr[2]}|${common_after_arr[2]}|g" "$path"/mkdocs/mindocr_data_readme.md
  sed -i "s|${before_arr[3]}|${common_after_arr[3]}|g" "$path"/mkdocs/customize_model.md
  for var in $(seq 5 "$count")
  do
    sed -i "s|${before_arr[var-1]}|${common_after_arr[var-1]}|g" "$path"/README.md
  done
}

# common
common_after_arr=("./README.md"
"./mindocr_data_readme.md"
"(customize_data_transform.md)"
"./mindocr_models_readme.md"
"("
"(mkdocs/online_inference.md)"
"(mkdocs/customize_dataset.md)"
"(mkdocs/customize_data_transform.md)"
"(mkdocs/customize_model.md)"
"(mkdocs/customize_postprocess.md)"
"(mkdocs/contributing.md)"
"(mkdocs/license.md)"
"](https://github.com/mindspore-lab/mindocr/blob/main/configs"
"(https://github.com/mindspore-lab/mindocr/blob/main/deploy/"
)

# en readme replace path
en_before_arr=("../../README.md"
"../../../mindocr/data/README.md"
"(transforms/README.md)"
"../../../mindocr/models/README.md"
"(docs/en/"
"(tools/infer/text/README.md)"
"(mindocr/data/README.md)"
"(mindocr/data/transforms/README.md)"
"(mindocr/models/README.md)"
"(mindocr/postprocess/README.md)"
"(CONTRIBUTING.md)"
"(LICENSE)"
"](configs"
"](deploy/")

ln -s "$PWD"/README.md "$EN_PATH"/README.md
ln -s "$PWD"/mindocr/data/README.md "$EN_PATH"/mkdocs/mindocr_data_readme.md
ln -s "$PWD"/mindocr/models/README.md "$EN_PATH"/mkdocs/mindocr_models_readme.md

replace_path "$EN_PATH" "${en_before_arr[*]}"

# zh readme replace path
zh_before_arr=("../../README_CN.md"
"../../../mindocr/data/README_CN.md"
"(transforms/README_CN.md)"
"../../../mindocr/models/README_CN.md"
"(docs/zh/"
"(tools/infer/text/README_CN.md)"
"(mindocr/data/README_CN.md)"
"(mindocr/data/transforms/README_CN.md)"
"(mindocr/models/README_CN.md)"
"(mindocr/postprocess/README_CN.md)"
"(CONTRIBUTING_CN.md)"
"(LICENSE)"
"](configs"
"](deploy/")

ln -s "$PWD"/README_CN.md "$ZH_PATH"/README.md
ln -s "$PWD"/mindocr/data/README_CN.md "$ZH_PATH"/mkdocs/mindocr_data_readme.md
ln -s "$PWD"/mindocr/models/README_CN.md "$ZH_PATH"/mkdocs/mindocr_models_readme.md

replace_path "$ZH_PATH" "${zh_before_arr[*]}"
