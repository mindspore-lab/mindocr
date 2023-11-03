### Q1 缺少符号

- `undefined symbol:_ZN9mindspore5tracel15GetDebugInfostrERKSt10shared_ptrINS_9DebugInfoEERKSsNS_13SourceLineTipE`

  ```bash
  Python 3.7.16 (default, Jan 17 2023, 22:20:44)	
  [GCC 11.2.0] :: Anaconda, Inc. on linux
  Type "help", "copyright", "credits" or "license" for more infommation.
  >>> import mindspore
  >>> import mindspore lite
  Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  	File "/root/miniconda3/envs/xxx/lib/python3.7/site-packages/mindspore_lite/_init_.py", line 26, in <module>	
  		from mindspore lite.context import Context
  	File "/root/miniconda3/envs/xxx/lib/python3.7/site-packages/mindspore_lite/context.py", line 22, in <module>
  		from mindspore lite.lib import-c lite wrapper	
  ImportError: xxxx/mindspore-lite-2.2.0-linux-x64/tools/converter/lib/libmindspore_converter.so: undefined symbol:_ZN9mindspore5tracel15GetDebugInfostrERKSt10shared_ptrINS_9DebugInfoEERKSsNS_13SourceLineTipE
  ```

- `undefined symbol: _ZN9mindspore12label_manage23GetGlobalTraceLabelTypeEv`

  ```bash
  Python 3.7.16 (default, Jan 17 2023, 22:20:44)	
  [GCC 11.2.0] :: Anaconda, Inc. on linux
  Type "help", "copyright". "credits" or "license" for more infommation.
  >>> import mindspore_lite
  >>> import mindspore
  Traceback (most recent call last):
  File "<stdin>", line 1, in <module>	
  	File "/ root/miniconda3/envs/xxx/1ib/python3.7/site-packages/mindspore/_init_.py", line 18, in <module>
  		from mindspore.run check import run check	
  	File "/root/miniconda3/envs/xxx/lib/python3.7/site-packages/mindspore/ run_check/_init_.py", line 17, in <module>
  		from . check_version import check_version_and_env_config
  	File "/root/miniconda3/envs/xxx/lib/python3.7/site-packages/mindspo re/run_check/check version.py", line 29, in <module>
  		from mindspore._c_expression "import MSContext, ms_ctx_param
  ImportError: xxxx/mindspore-lite-2.2.0-linux-x64/tools/converter/lib/libmindspore_converter.so: undefined symbol: _ZN9mindspore12label_manage23GetGlobalTraceLabelTypeEv
  ```

- `undefined symbol: _ZNK9mindspore6kernel15KernelBuildInfo8TostringEv`

  ```bash
  [WARNING] LITE(20788, 7f897f04ff40, converter_lite) :2023-10-19-07:24: 10.858.973 [mindspore/lite/tools/opt imizer/common/fommat_utils.cc:385] ConvertAbstractFommatShape] abstract must be a tensor, but got: ValueAny.	
  [WARNING] LITE(20788,7f897f04ff40, converter_lite) :2023-10-19-07:24: 10.858.998 [mindspore/lite/tools/optimizer/common/gllo_utils.cc: 1071] GenTransposeNode] Convertabstract failed for node: args0_nh2nc	
  [WARNING] LITE(20788,7f897fO04ff40, converter_lite) :2023-10-19-07:24: 11.035.069 [mindspore/lite/src/extendrt/cxx_api/dlutils.h:124] DLSo0pen] dlopen /xxx/mindspore/to0ls/converter/lib/libascend pass plugin.so failed, error: /xxx/mindspore/tools/converter/1ib/libmslite_shared lib.s0: undefined symbol: _ZNK9mindspore6kernel15KernelBuildInfo8TostringEv
  
  [ERROR] LITE(20788,7f897f04ff40, converter_lite) :2023-10-19-07:24: 11.035.121 [mindspore/lite/tools/converter/adapter/acl/plugin/acl_pass_plugin.cc:86] CreateAclPassInner] DLSo0pen failed, so path: /xxx/mindspore-1ite-2.2.0.20231019-1inux-x64/tools/converter/lib/1ibascend_pass_plugin.so, ret: dlopen /xxx/mindspore/tools/converter/lib/libascend_pass_plugin.so failed, error: /xxx/mindspore/tools/converter/lib/libmslite shared lib.so: undefined symbol: _ZNK9mindspore6kernel15KernelBuildInfo8TostringEv
  ```


以上缺少符号问题，为`mindspore` python whl包、`mindspore_lite` python whl包、`mindspore_lite`tar包不匹配导致。根据[下载mindspore](https://www.mindspore.cn/install) 和 [下载MindSpore Lite](https://www.mindspore.cn/lite/docs/zh-CN/master/use/downloads.html)，需检查

- `mindspore`, `mindspore_lite`是否版本一致，例如都为2.2.0版本；
- `mindspore_lite`的whl包与`mindspore_lite`的tar包是否版本一致，例如都为2.2.0版本；
- `mindspore_lite`的whl包与`mindspore_lite`的tar包是否都为云侧版本

例如平台为linux x86_64下的昇腾环境，如下包组合是合适的

- `mindspore_lite whl`: [mindspore_lite-2.2.0-cp37-cp37m-linux_x86_64.whl](https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.2.0/MindSpore/lite/release/linux/x86_64/cloud_fusion/python37/mindspore_lite-2.2.0-cp37-cp37m-linux_x86_64.whl)
- `mindspore_lite tar.gz`: [mindspore-lite-2.2.0-linux-x64.tar.gz](https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.2.0/MindSpore/lite/release/linux/x86_64/cloud_fusion/python37/mindspore-lite-2.2.0-linux-x64.tar.gz)
- `mindspore whl`: [mindspore-2.2.0-cp37-cp37m-linux_x86_64.whl](https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.2.0/MindSpore/unified/x86_64/mindspore-2.2.0-cp37-cp37m-linux_x86_64.whl)

### Q2 Ascend so库找不到相关错误

- `dlopen mindspore_lite/lib/libascend_kernel_plugin.so` ，No such file or directory

  **错误信息**
  
  ```bash
  File "/home/xxx/miniconda3/envs/yyy/lib/python3.8/site-packages/mindspore_lite/model.py", line 95, in warpper
  	return func(*args, **kwargs)
  File "/home/xxx/miniconda3/envs/yyy/lib/python3.8/site-packages/mindspore_lite/model.py", line 235, in build_from_file
  	raise RuntimeError(f"build from_file failed! Error is {ret.Tostring()}")
  RuntimeError: build_from_file failed! Error is Common error code.    
  [WARNING] ME(15411,7f07f56be100, python) : 2023-10-16-00:51:42.509.780 [mindspore/lite/src/extend rt/cxx_api/dlutils.h:124] DLSo0pen] 
  dlopen /home/xxx/miniconda3/envs/yyy/lib/python3.8/site-packages/mindspore_lite/lib/libascend_kernel_plugin.so failed, error: libacl_cblas.so: cannot open shared object file: No such file or directory
  [ERROR] ME(15411,7f07f56be100, python) :2023-10-16-00:51:42.509.877 [mindspo re/lite/src/extendrt/kernel/ascend/plugin/ascend_allocator_plugin.cc:70] Register] DLSo0pen failed, so path: /home/xxx/miniconda3/envs/ yyy/lib/python3.8/site-packages/mindspore_lite/lib/libascend_kernel_plugin.so , func name: CreateAclAllocator. err: dlopen /home/xxx/miniconda3/envs/yyy/lib/python3.8/site-packages/mindspore_lite/lib/libascend_ kernel_plugin.so failed, error: libacl_cblas.so: cannot open shared object file: No such file or directory    
  [ERROR] ME(15411,7f07f56be100, python):2023-10-16-00:51:42.509.893 [mindspore/lite/src/extendrt/infer_session.cc:66] HandleContext] failed register ascend allocator plugin.    
  ...
  raise RuntimeError(f"build_from_file failed! Error is {ret.ToString()}")
  RuntimeError: build_from_file failed! Error is Common error code.
  ```
  
  **解决方法**
  
  该错误是`mindspore_lite`tar包中的`libascend_kernel plugin.so`未加入到环境变量`LD_LIBRARY_PATH`导致，解决方法如下
  
  1. 查看是否安装了`mindspore_lite`的**云侧推理工具包**。如果未安装，请从 [工具包tar.gz、whl包下载链接](https://gitee.com/link?target=https%3A%2F%2Fwww.mindspore.cn%2Flite%2Fdocs%2Fzh-CN%2Fmaster%2Fuse%2Fdownloads.html)，下载Ascend版的云侧版本`tar.gz`包以及`whl`包安装，详细请见 [mindspore lite 安装](https://gitee.com/mindspore-lab/mindocr/blob/main/docs/cn/inference/environment.md)。
  
  2. 找到`mindspore_lite`的安装路径，如路径为`/your_path_to/mindspore-lite`，cd到该目录下

  3. 查找`libascend_kernel_plugin.so`，命令为`find ./ -name libascend_kernel_plugin.so`，可以找到该so的路径为

  ```bash
  ./runtime/lib/libascend_kernel_plugin.so
  ```

  4. 将该路径加入到环境变量

  ```bash
  export LD_LIBRARY_PATH=$LITE_HOME/runtime/lib:$LD_LIBRARY_PATH
  ```
  
- `Load dynamic library: libmindspore_ascend.so.2 failed. liboptiling.so: cannot open shared object file: No such file or directory`

  ```bash
  python -c "import mindspore;mindspore.set_context(device_target='Ascend');mindspore.run_check()"
  [WARNING] ME(60105:13981374421 1776, MainProcess):2023-10-25-08: 14:33.640.411 [mindspore/run_check/_check_version.py:348] Using custom Ascend AI software package (Ascend Data Center Solution) path, package version checking is skipped. Please make sure Ascend AI software package (Ascend Data Center Solution) version is supported. For details, refer to the installation guidelines https://www.mindspore.cn/install
  Traceback (most recent call last):
  File "<string>", line 1, in module>
  File "/xxx/py37/lib/python3.7/site-packages/mindspore/_checkparam.py", line 1313, in wrapper	
  	return func(*args, **kwargs)
  File "/xxx/py37/1ib/python3.7/site-packages/mindspore/context.py", line 1456, in set_context
  	ctx.set_device_target(kwargs['device target'])
  File "/xxx/py37/lib/python3.7/site-packages/mindspore/context.py", line 381, in set_device_target
  	self.set_param(ms_ctx_param.device_target, target)
  File "/xxx/py37/lib/python3.7/site-packages/mindspore/context.py", line 175, in set_param
  	self._context_handle.set_param(param, value)
  RuntimeError: Unsupported device target Ascend. This process only supports one of the ['CPU']. Please check whether the Ascend environment is installed and configured correctly. and check whether current mindspore wheel package was built with "-e Ascend". For details, please refer to "Device load error message".
  
  ----------------------------------------------------
  - Device load error message:
  ----------------------------------------------------
  Load dynamic library: libmindspore_ascend.so.2 failed. liboptiling.so: cannot open shared object file: No such file or directory
  Load dynamic library: 1ibmindspore ascend.so.1 failed. liboptiling.so: cannot open shared object file: No such file or directory	
  
  ----------------------------------------------------
  ...
  ```

  该错误为`liboptiling.so`未加入环境变量`LD_LIBRARY_PATH`导致，解决方法如下

  1. 查看是否安装了`CANN`。如果未安装，请[安装昇腾AI处理器配套软件包](https://www.mindspore.cn/install#%E5%AE%89%E8%A3%85%E6%98%87%E8%85%BEai%E5%A4%84%E7%90%86%E5%99%A8%E9%85%8D%E5%A5%97%E8%BD%AF%E4%BB%B6%E5%8C%85)，安装CANN。

  2. 找到`CANN`的安装路径，如路径为`/your_path_to/cann`，cd到该目录下

  3. 查找`liboptiling.so`，命令为`find ./ -name liboptiling.so`，可以找到该so的路径为

     ```bash
     ./CANN-7.0/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64/liboptiling.so
     ./CANN-7.0/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/liboptiling.so
     ./CANN-7.0/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/minios/aarch64/liboptiling.so
     ```

  4. 将该路径加入到环境变量，若为`x86_64`，则

     ```bash
     export LD_LIBRARY_PATH=$ASCEND_HOME/CANN-7.0/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64/:$LD_LIBRARY_PATH
     ```
  
     当下列信息出现时，提示安装成功
  
     ```bash
     The result of multiplication calculation is correct. MindSpore has been installed on platform [Ascend] successfully!
     ```



- `Load dynamic library: libmindspore_ascend.so.2 failed. libaicpu_ascend_engine.so: cannot open shared object file: No such file or directory`

  ```bash
  RuntimeError: Unsupported device target Ascend. This process only supports one of the ['CPU']. Please check whether the Ascend environment is installed and configured correctly. and check whether current mindspore wheel package was built with "-e Ascend". For details, please refer to "Device load error message".
  
  ----------------------------------------------------
  - Device load error message:
  ----------------------------------------------------
  Load dynamic library: libmindspore_ascend.so.2 failed. libaicpu_ascend_engine.so: cannot open shared object file: No such file or directory
  Load dynamic library: libmindspore_ascend.so.1 failed. libaicpu_ascend_engine.so: cannot open shared object file: No such file or directory		
  
  ----------------------------------------------------
  ...
  
  ```

  该错误为`libaicpu_ascend_engine.so`未加入环境变量`LD_LIBRARY_PATH`导致，解决方法如下

  1. 查看是否安装了`CANN`。如果未安装，请[安装昇腾AI处理器配套软件包](https://www.mindspore.cn/install#%E5%AE%89%E8%A3%85%E6%98%87%E8%85%BEai%E5%A4%84%E7%90%86%E5%99%A8%E9%85%8D%E5%A5%97%E8%BD%AF%E4%BB%B6%E5%8C%85)，安装CANN。

  2. 找到`CANN`的安装路径，如路径为`/your_path_to/cann`，cd到该目录下

  3. 查找`libaicpu_ascend_engine.so`，命令为`find ./ -name libaicpu_ascend_engine.so`，可以找到该so的路径为

     ```bash
     ./CANN-7.0/x86_64-linux/lib64/plugin/opskernel/libaicpu_ascend_engine.so
     ./CANN-7.0/compiler/lib64/plugin/opskernel/libaicpu_ascend_engine.so
     ./latest/x86_64-linux/lib64/plugin/opskernel/libaicpu_ascend_engine.so
     ```

  4. 将路径加入到环境变量，若为`x86_64`，则

     ```bash
     export LD_LIBRARY_PATH=$ASCEND_HOME/CANN-7.0/compiler/lib64/plugin/opskernel/:$LD_LIBRARY_PATH
     ```

  

### Q3 Ascend Error Message A39999

- 错误1

  ```bash
  ----------------------------------------------------
  - Ascend Error Message:
  ----------------------------------------------------
  E39999: Inner Error!	
  E39999 TsdOpen failed. devId=0, tdt error=31[FUNC:PrintfTsdError] [FILE: runtime.cc][LINE:2060]
  	   TraceBack (most recent call last):
  	   Start aicpu executor failed, retCode=0x7020009 devId=0[FUNC :DeviceRetain][FILE: runtime.cc][LINE:2698]
  	   check param failed, dev can not be NULL![FUNC:PrimaryContextRetain][FILE: runtime.cc][LINE:2544]
  	   Check param failed, ctx can not be NULL! [FUNC:PrimaryContextRetain][FILE: runtime.cc][LINE:2571]
  	   Check param failed, context can not be null.[FUNC:NewDevice][FILE:api impl.cc][LINE:1899]	
  	   New device failed, retcode=0x70 10006[FUNC:SetDevice][FILE:api_impL-cc][LINE:1922]	
  	   rtsetDevice execute failed, reason=[device retain error][FUNC:FuncErrorReason][FILE :error message manage.ccl[LINE:50]
  	   open device 0 failed runtime result = 507033.[FUNC: ReportCallError][FILE:log_inner.cpp][LINE:161]
  
  (Please search "Ascend Error Message" at https://www.mindspore.cn for error code description)
  ```

- 错误2

  ```bash
  ----------------------------------------------------
  - Ascend Error Message:
  ----------------------------------------------------
  E39999: Inner Error!
  E39999 tsd client wait response fail, device response code[1]. load aicpu ops package failed, device[O], host pid[5653], error stack:
  [TSDaemon] checksum aicpu package failed, ret=103, [tsd_common.cpp:2242:SaveProcessConfig]17580
  Check head tag failed, ret=279, [package_worker.cpp:537:VerifyAicpuPackage]2369
  Verify Aicpu package failed, srcPathlIhome/HMHiAiuser/aicpu_kernels/vf0_5653_Ascend310P-aicpu_syskernels.tar.gz]..[package_worker.cpp:567:DecompressionAicpuPackage]2369
  Decompression AicpuPackage [/home/HwHiAiUser/aicpu_kernels/vf0_5653_Ascend310P-aicpu_syskernels.tar.gz] failed, [package_worker.cpp:218:LoadAICPUPackageForProcessMode]2369
  Load aicpu package path[/home/HwHiAiUser/hdcd/device0/] fileName[ 5653_Ascend310P-aicpu_syskernels.tar.gz] failed, [inotify_watcher.cpp:311:HandleEvent]2369
  [TSDaemon] load aicpu ops package failed, device[0], host pid[5653], [tsd_common.cpp:2054:CheckAndHandleTimeout]2374
  [FUNC:WaitRsp][FILE:process_mode_manager.cpp][LINE:270]
  	TraceBack (most recent call last):
  	TsdOpen failed. devId=0, tdt error=31[FUNC:PrintfTsdError] [FILE: runtime.cc][LINE:2060]
  	Start aicpu executor failed, retCode=Ox7020009 devId=0[FUNC:DeviceRetain][FILE: runtime.cc] [LINE:2698]
  	Check param failed, dev can not be NULL! [FUNC:PrimaryContextRetain] [FILE: runtime.cc][LINE:2544]
  	Check param failed, ctx can not be NULL! [FUNC:PrimaryContextRetain J[FILE: runtime.cc][LINE:2571]
  	Check param failed, context can not be null. [FUNC:NewDevice] [FILE:api_impl.cc][LINE: 1893]
  	New device failed, retCode=0x7010006[FUNC:SetDevice] [FILE:api impl.cc][LINE:1916]
  	rtSetDevice execute failed, reason=[device retain errorl[FUNC:FuncErrorReason] [FILE:error_message_manage.cc][LINE:50]
  	open device 0 failed, runtime result = 507033. [FUNC:ReportCalLError][FILE:log_inner.cpp]ILINE:161]
  (Please search "Ascend Error Message" at https://www.mindspore.cn for error code description)
  ```

可能原因

- 驱动版本与CANN不匹配

- 环境变量未配置成功，导致aicpu启动失败，尝试将下列项加入环境变量

  ```bash
  export ASCEND_OPP_PATH=${ASCEND_HOME}/latest/opp
  export ASCEND_AICPU_PATH=${ASCEND_OPP_PATH}/..
  ```

  

### Q4 `acl open device 0 failed`

推理时可能触发`acl open device 0 failed`，例如

```bash
benchmark --modelFile=dbnet_mobilenetv3_lite.mindir --device=Ascend --inputShapes='1,3,736,1280' --loopCount=100 - -wammUpLoopCount=10
ModelPath = dbnet_mobilenetv3_lite.mindir	
ModelType = MindIR
InDatapath =
GroupInfoFile =
ConfigFilepath =
InDataType = bin
LoopCount = 100
DeviceType = Ascend
AccuracyThreshold = 0.5
CosineDistanceThreshold = -1.1
WarmUpLoopCount = 10
NumThreads = 2	InterOpParallelNum = 1
Fpl16Priority = 0	EnableparalÍel = 0
calibDataPath =
EnableGLTexture = 0
cpuBindMode = HIGHER CPU
CalibDataType = FLOAT
Resize Dims: 1 3 736 1280
start unified benchmark run
IERROR] ME (26748,7f6c73867fc0, benchmark) :2023-10-26-09:51 : 54.833.515 Imindspore/lite/src/extend rt/kernel/ascend/model/model_infer.cc:59] Init] Acl open device 0 failed.
[ERROR] ME (26748,7f6c73867fc0,benchmark):2023-10-26-09:51:54.833.573 [mindspore/lite/src/extend rt/kernel/ascend/src/custom_ascend_kernel.cc:141] Init] Model i
nfer init failed.	[ERROR] ME (26748, 7f6c73867fc0,benchmark) :2023-10-26-09:51:54.833.604 [mindspore/lite/src/extendrt/session/single_op_session.cc:198] BuildCustomAscendKernelImpl] kernel init failed CustomAscend
[ERROR] ME (26748,7f6c73867fc0, benchmark) :2023-10-26-09:51 :54.833.669 [mindspore/li te/src/extendrt/session/single_op_sess ion.cc:220] BuildCustomAscendKernel] Build ascend kernel failed for node: custom_0
[ERROR] ME (26748,7f6c73867fc0,benchmark) :2023-10-26-09:51 : 54.833.699 [mindspore/lite/src/extend rt/session/single_op_session.cc:302] CompileGraph] Failed to Build custom ascend kernel
[ERROR] ME (26748,7f6c73867fc0,benchmark) :2023-10-26-09:51:54.833.727 [mindspore/lite/s rc/extendrt/cxx_api/model/model_impl.cc:413] BuildByBufferImpl] compile graph failed.
[ERROR] ME (26748, 7f6c73867fc0, benchmark):2023-10-26-09:51:54.835.590 [mindspore/lite/tools/benchma rk/benchmark_unified_api.cc:1256] CompileGraph] ms_model_.Build failed while running
IERROR] ME (26748,7f6c73867fc0,benchmark) :2023-10-26-09:51:54.835.627 [mindspore/lite/tools/benchma rk/benchmark_unified_api.cc:1325] RunBenchmark] Compile graph failed.
[ERROR] ME(26748,7f6c73867fc0, benchmark):2023-10-26-09: 51:54.835.662 [mindspore/lite/tools/benchmark/ run_benchmark.cc :78] RunBenchmark] Run Benchmark dbnet_mobilenetv3_lite.mindi r Failed : -1
ms_model_.Build failed while running Run Benchmark dbnet mobilenetv3 lite.mindir Failed : -1
```

环境变量缺少配置`acllib`相关库

```bash
export NPU_HOST_LIB=$ASCEND_HOME/latest/acllib/lib64/stub
export DDK_PATH=$ASCEND_HOME/latest
export LD_LIBRARY_PATH=$ASCEND_HOME/latest/acllib/lib64
export ASCEND_AICPU_PATH=$ASCEND_HOME/latest/x86_64-linux
export LD_LIBRARY_PATH=$ASCEND_HOME/latest/x86_64-linux/lib64:$LD_LIBRARY_PATH
```



### Q5 windows安装mindocr依赖失败

windows下执行

```bash
git clone git@gitee.com:mindspore-lab/mindocr.git
cd mindocr
pip install -e .
```

报`lanms`安装失败，错误信息为

```bash
FileNotFoundError: [WinError 2] 系统找不到指定的文件。
```

`lanma`似乎不支持windows，[https://github.com/argman/EAST/](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2Fargman%2FEAST%2F) 上仍存在适配windows的issue。推荐使用linux。
用windows可考虑用`lanms-neo`替换`lanms`并安装。安装过程可能遇到以下错误

```bash
Looking in indexes: https://pypi.tuna.tsinghua.edu.cn/simple
Collecting lanms-neo==1.0.2
  Downloading https://pypi.tuna.tsinghua.edu.cn/packages/7b/fe/beff7e7e4455cb9f69c5734897ca8552a57f6423b062ec86b2ebc1d79c0d/lanms_neo-1.0.2.tar.gz (39 kB)
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Preparing metadata (pyproject.toml) ... done
Building wheels for collected packages: lanms-neo
  Building wheel for lanms-neo (pyproject.toml) ... error
  error: subprocess-exited-with-error

  × Building wheel for lanms-neo (pyproject.toml) did not run successfully.
  │ exit code: 1
  ╰─> [10 lines of output]
      running bdist_wheel
      running build
      running build_py
      creating build
      creating build\lib.win-amd64-cpython-37
      creating build\lib.win-amd64-cpython-37\lanms
      copying lanms\__init__.py -> build\lib.win-amd64-cpython-37\lanms
      running build_ext
      building 'lanms._C' extension
      error: Microsoft Visual C++ 14.0 or greater is required. Get it with "Microsoft C++ Build Tools": https://visualstudio.microsoft.com/visual-cpp-build-tools/
      [end of output]

  note: This error originates from a subprocess, and is likely not a problem with pip.
  ERROR: Failed building wheel for lanms-neo
Failed to build lanms-neo
ERROR: Could not build wheels for lanms-neo, which is required to install pyproject.toml-based projects
```

需 [安装开发工具](https://visualstudio.microsoft.com/zh-hans/visual-cpp-build-tools/)。并执行`pip install lanms-neo`。

删除`requirement.txt`中的`lanms`依赖项，继续执行`pip install -r requirements.txt`可完成安装。



### Q6 `RuntimeError: The device address type is wrong: type name in address:CPU, type name in context:Ascend`

- export时触发 `RuntimeError: The device address type is wrong: type name in address:CPU, type name in context:Ascend`，例如

  ```bash
  [WARNING] ME(18680:139900608063296,MainProcess):2023-10-31-12:31:20.141.25 [mindspore/run_check/_check_version.py:348] Using custom Ascend AI software package (Ascend Data Center Solution) path, package version checking is skipped. Please make sure Ascend AI software package (Ascend Data Center Solution) version is supported. For details, refer to the installation guidelines https://www.mindspore.cn/install
  [WARNING] ME(18680:139900608063296,MainProcess):2023-10-31-12:31:20.143.96 [mindspore/run_check/_check_version.py:460] Can not find the tbe operator implementation(need by mindspore-ascend). Please check whether the Environment Variable PYTHONPATH is set. For details, refer to the installation guidelines: https://www.mindspore.cn/install
  [WARNING] ME(18680:139900608063296,MainProcess):2023-10-31-12:31:20.144.71 [mindspore/run_check/_check_version.py:466] Can not find driver so(need by mindspore-ascend). Please check whether the Environment Variable LD_LIBRARY_PATH is set. For details, refer to the installation guidelines: https://www.mindspore.cn/install
  Traceback (most recent call last):
    File "tools/export.py", line 173, in <module>
      export(**vars(args))
    File "tools/export.py", line 73, in export
      net = build_model(model_cfg, pretrained=True, amp_level=amp_level)
    File "/xxx/mindocr/mindocr/models/builder.py", line 52, in build_model
      network = create_fn(**kwargs)
    File "/xxx/mindocr/mindocr/models/rec_svtr.py", line 122, in svtr_tiny_ch
      model = SVTR(model_config)
    File "/xxx/mindocr/mindocr/models/rec_svtr.py", line 26, in __init__
      BaseModel.__init__(self, config)
    File "/xxx/mindocr/mindocr/models/base_model.py", line 34, in __init__
      self.backbone = build_backbone(backbone_name, **config.backbone)
    File "/xxx/mindocr/mindocr/models/backbones/builder.py", line 48, in build_backbone
      backbone = backbone_class(**kwargs)
    File "/xxx/mindocr/mindocr/models/backbones/rec_svtr.py", line 486, in __init__
      ops.zeros((1, num_patches, embed_dim[0]), ms.float32)
    File "/xxx/py37/lib/python3.7/site-packages/mindspore/ops/function/array_func.py", line 1039, in zeros
      output = zero_op(size, value)
    File "/xxx/py37/lib/python3.7/site-packages/mindspore/ops/primitive.py", line 314, in __call__
      return _run_op(self, self.name, args)
    File "/xxx/py37/lib/python3.7/site-packages/mindspore/ops/primitive.py", line 913, in _run_op
      stub = _pynative_executor.run_op_async(obj, op_name, args)
    File "/xxx/py37/lib/python3.7/site-packages/mindspore/common/api.py", line 1186, in run_op_async
      return self._executor.run_op_async(*args)
  RuntimeError: The device address type is wrong: type name in address:CPU, type name in context:Ascend
  
  ----------------------------------------------------
  - C++ Call Stack: (For framework developers)
  ----------------------------------------------------
  mindspore/ccsrc/plugin/device/ascend/hal/hardware/ge_device_res_manager.cc:72 AllocateMemory
  ```

- MindSporeAscend模式下进行计算发生错误。触发`RuntimeError: The device address type is wrong: type name in address:CPU, type name in context:Ascend`，例如

  ```python
  Python 3.7.16 (default, Jan 17 2023, 22:20:44) 
  [GCC 11.2.0] :: Anaconda, Inc. on linux
  Type "help", "copyright", "credits" or "license" for more information.
  >>> import numpy as np
  >>> import mindspore as ms
  [WARNING] ME(44720:140507814819648,MainProcess):2023-11-01-03:01:38.884.384 [mindspore/run_check/_check_version.py:348] Using custom Ascend AI software package (Ascend Data Center Solution) path, package version checking is skipped. Please make sure Ascend AI software package (Ascend Data Center Solution) version is supported. For details, refer to the installation guidelines https://www.mindspore.cn/install
  [WARNING] ME(44720:140507814819648,MainProcess):2023-11-01-03:01:38.884.675 [mindspore/run_check/_check_version.py:466] Can not find driver so(need by mindspore-ascend). Please check whether the Environment Variable LD_LIBRARY_PATH is set. For details, refer to the installation guidelines: https://www.mindspore.cn/install
  >>> import mindspore.ops as ops
  >>> ms.set_context(device_target="Ascend")
  >>> ms.run_check()
  MindSpore version:  2.2.0.20231025
  The result of multiplication calculation is correct, MindSpore has been installed on platform [Ascend] successfully!
  >>> x = ms.Tensor(np.ones([1,3,3,4]).astype(np.float32))
  >>> y = ms.Tensor(np.ones([1,3,3,4]).astype(np.float32))
  >>> print(ops.add(x, y))
  Traceback (most recent call last):
    File "<stdin>", line 1, in <module>
    File "/root/miniconda3/envs/zhqtest/lib/python3.7/site-packages/mindspore/common/_stub_tensor.py", line 49, in fun
      return method(*arg, **kwargs)
    File "/root/miniconda3/envs/zhqtest/lib/python3.7/site-packages/mindspore/common/tensor.py", line 493, in __str__
      return str(self.asnumpy())
    File "/root/miniconda3/envs/zhqtest/lib/python3.7/site-packages/mindspore/common/tensor.py", line 964, in asnumpy
      return Tensor_.asnumpy(self)
  RuntimeError: The device address type is wrong: type name in address:CPU, type name in context:Ascend
  
  ----------------------------------------------------
  - C++ Call Stack: (For framework developers)
  ----------------------------------------------------
  mindspore/ccsrc/plugin/device/ascend/hal/hardware/ge_device_res_manager.cc:72 AllocateMemory
  ```

  

上述问题，为 Ascend 310/Ascend 310P3 下可能不支持MindSporeAscend模式下的计算。请到[下载mindspore](https://www.mindspore.cn/install)确认安装的版本是否与mindspore版本和硬件平台配套，或这你可以

- 将MindSpore模式调为CPU
- 使用 [MindSpore Lite](https://www.mindspore.cn/lite)
