## Windows C++推理
### 环境配置
1. 下载[GCC](https://sourceforge.net/projects/mingw-w64/files/Toolchains%20targetting%20Win64/Personal%20Builds/mingw-builds/7.3.0/threads-posix/seh/x86_64-7.3.0-release-posix-seh-rt_v5-rev0.7z/download)并解压；
2. 将GCC解压后的目录```mingw64/bin```加入到环境变量Path里；
3. 下载[CMake](https://github.com/Kitware/CMake/releases/download/v3.18.3/cmake-3.18.3-win64-x64.msi)安装，在安装过程中注意勾选Add CMake to the system PATH for the current user，将cmake添加到Path环境变量：
<div align=center>
<img width="388" alt="WechatIMG92" src="https://github.com/VictorHe-1/DDPM/assets/80800595/e59db28d-330e-478a-b38a-3aab3ac7248c"></div>

4. 下载[MindSpore Lite](https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.0.0/MindSpore/lite/release/windows/mindspore-lite-2.0.0-win-x64.zip)并解压；
5. 下载[opencv-mingw预编译3.4.8 x64版本](https://github.com/huihut/OpenCV-MinGW-Build/archive/refs/tags/OpenCV-3.4.8-x64.zip)，并解压；
6. 将OpenCV解压后的```x64/mingw/bin```路径加入到环境变量Path里；
6. 下载[MindOCR代码](https://codeload.github.com/liangxhao/mindocr/zip/refs/heads/cpp_infer)并解压；
7. 下载[Clipper](https://udomain.dl.sourceforge.net/project/polyclipping/clipper_ver6.4.2.zip)并解压，将```cpp```目录下的```clipper.cpp```和```clipper.hpp```文件拷贝到MindOCR代码目录```deploy/cpp_infer_ddl/src/data_process/postprocess```里:
<div align=center>
<img width="574" alt="WechatIMG91" src="https://github.com/VictorHe-1/DDPM/assets/80800595/5d243531-2f7f-4c34-bbb9-060f93bf0d92"></div>

<div align=center>
<img width="551" alt="WechatIMG93" src="https://github.com/VictorHe-1/DDPM/assets/80800595/cffb644d-2bad-46dd-92c7-24d190759e88"></div>

8. 打开[下载页面](https://download.mindspore.cn/toolkits/mindocr/windows/)：
<div align=center>
<img width=80% alt="f0" src="https://github.com/VictorHe-1/DDPM/assets/80800595/8b748e18-d1b5-4023-9875-c8f9ef44036e"></div>

下载数据集[ic15.zip](https://download.mindspore.cn/toolkits/mindocr/windows/ic15.zip)，[文本检测模型](https://download.mindspore.cn/toolkits/mindocr/windows/ch/ch_ppocr_server_v2.0_det_infer_cpu.ms)，[文本识别模型](https://download.mindspore.cn/toolkits/mindocr/windows/ch/ch_ppocr_server_v2.0_rec_infer_argmax_cpu.ms)以及[字典文件](https://download.mindspore.cn/toolkits/mindocr/windows/ch/ppocr_keys_v1.txt)。
<br></br>

### 推理方法
1. 进入MindOCR代码目录```deploy/cpp_infer_ddl/src/```，修改```build.bat```中的MindSpore Lite路径以及OpenCV路径，示例如下：
```text
set LITE_HOME=D:\mindocr_windows\mindspore-lite-2.0.0-win-x64
set OPENCV_DIR=D:\mindocr_windows\OpenCV-MinGW-Build-OpenCV-3.4.8-x64
```
**注意：在修改MindSpore_lite路径```LITE_HOME```和OpenCV路径```OPENCV_DIR```时，需要写成上述样例的反斜杠```\```写法。**

2. 运行```build.bat```文件（双击打开或者命令行里输入```build.bat```并回车），等待编译完成后，在```deploy/cpp_infer_ddl/src/dist```目录下会生成```infer.exe```文件；

3. build完成后使用```deploy/cpp_infer_ddl/src/infer.bat```进行推理，注意修改infer.bat里的以下参数：
```text
LITE_HOME=D:/mindocr_windows/mindspore-lite-2.0.0-win-x64 # mindspore lite路径

OPENCV_DIR=D:/mindocr_windows/OpenCV-MinGW-Build-OpenCV-3.4.8-x64 # OpenCV路径

--input_images_dir D:\ic15\det\test\ch4_test_images # 测试图片目录
--det_model_path D:\models\ch_ppocr_server_v2.0_det_infer_cpu.ms # 文本检测模型目录
--rec_model_path D:\models\ch_ppocr_server_v2.0_rec_infer_argmax_cpu.ms # 文本识别模型目录
--character_dict_path D:\dict\ic15_dict.txt # 字典文件目录
```

**注意: ```LITE_HOME```和```OPENCV_DIR```需要设置成正斜杠```/```写法，infer.exe里面的路径参数都需要设置成反斜杠```\```，与如上样例保持一致**。

4. 在```deploy/cpp_infer_ddl/src/```目录中，打开cmd终端，使用以下命令执行推理:
```shell
infer.bat
```
5. 推理结果存在```deploy/cpp_infer_ddl/src/dist/det_rec```目录下；
