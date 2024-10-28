## Frequently Asked Questions
 - [Undefined symbol](#q1-undefined-symbol)
 - [Ascend so Not Found](#q2-ascend-so-not-found)
 - [Ascend Error Message A39999](#q3-ascend-error-message-a39999)
 - [acl open device 0 failed](#q4-acl-open-device-0-failed)
 - [Fail to install mindocr dependency on windows](#q5-fail-to-install-mindocr-dependency-on-windows)
 - [RunTimeError:The device address tpe is wrong](#q6-runtimeerror-the-device-address-type-is-wrong-type-name-in-addresscpu-type-name-in-contextascend)
 - [Problems related to model converting](#q7-problems-related-to-model-converting)
 - [Problems related to inference](#q8-problems-related-to-inference)
 - [Training speed of DBNet not as fast as expexted](#q9-training-speed-of-dbnet-not-as-fast-as-expexted)
 - [Error about libgomp-d22c30c5.so.1.0.0](#q10-error-about-libgomp-d22c30c5so100)
 - [Dataset Pipeline Error when training abinet on lmdb dataset](#q11-dataset-pipeline-error-when-training-abinet-on-lmdb-dataset)
 - [Runtime Error when training dbnet on synthtext dataset](#q12-runtime-error-when-training-dbnet-on-synthtext-dataset)
 - [Failed to install seqeval](#q13-failed-to-install-seqeval)
 - [Failed to install lanms](#q14-failed-to-install-lanms)


### Q1 Undefined symbol

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
  [WARNING] LITE(20788,7f897fO04ff40, converter_lite) :2023-10-19-07:24: 11.035.069 [mindspore/lite/src/extendrt/cxx_api/dlutils.h:124] DLSopen] dlopen /xxx/mindspore/to0ls/converter/lib/libascend pass plugin.so failed, error: /xxx/mindspore/tools/converter/1ib/libmslite_shared lib.s0: undefined symbol: _ZNK9mindspore6kernel15KernelBuildInfo8TostringEv

  [ERROR] LITE(20788,7f897f04ff40, converter_lite) :2023-10-19-07:24: 11.035.121 [mindspore/lite/tools/converter/adapter/acl/plugin/acl_pass_plugin.cc:86] CreateAclPassInner] DLSopen failed, so path: /xxx/mindspore-1ite-2.2.0.20231019-1inux-x64/tools/converter/lib/1ibascend_pass_plugin.so, ret: dlopen /xxx/mindspore/tools/converter/lib/libascend_pass_plugin.so failed, error: /xxx/mindspore/tools/converter/lib/libmslite shared lib.so: undefined symbol: _ZNK9mindspore6kernel15KernelBuildInfo8TostringEv
  ```


The problems occur due to the mismatch of `mindspore` python whl package, `mindspore_lite` python whl package and`mindspore_lite` tar package. Please check the following items according [MindSpore download](https://www.mindspore.cn/install/en) and [MindSpore Lite download](https://www.mindspore.cn/lite/docs/en/master/use/downloads.html):

- Consistent version of `mindspore`, `mindspore_lite`, e.g. 2.2.0
- Consistent version of `mindspore_lite` whl and `mindspore_lite` tar package, e.g. 2.2.0
- Both `mindspore_lite` whl package and `mindspore_lite` tar package are Cloud-side

For example, the following combination of packages is suitable when the platform is linux x86_64 with Ascend

- `mindspore_lite whl`: [mindspore_lite-2.2.0-cp37-cp37m-linux_x86_64.whl](https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.2.0/MindSpore/lite/release/linux/x86_64/cloud_fusion/python37/mindspore_lite-2.2.0-cp37-cp37m-linux_x86_64.whl)
- `mindspore_lite tar.gz`: [mindspore-lite-2.2.0-linux-x64.tar.gz](https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.2.0/MindSpore/lite/release/linux/x86_64/cloud_fusion/python37/mindspore-lite-2.2.0-linux-x64.tar.gz)
- `mindspore whl`: [mindspore-2.2.0-cp37-cp37m-linux_x86_64.whl](https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.2.0/MindSpore/unified/x86_64/mindspore-2.2.0-cp37-cp37m-linux_x86_64.whl)

### Q2 Ascend so Not Found

- `dlopen mindspore_lite/lib/libascend_kernel_plugin.so`, No such file or directory

  ```bash
  File "/home/xxx/miniconda3/envs/yyy/lib/python3.8/site-packages/mindspore_lite/model.py", line 95, in warpper
  	return func(*args, **kwargs)
  File "/home/xxx/miniconda3/envs/yyy/lib/python3.8/site-packages/mindspore_lite/model.py", line 235, in build_from_file
  	raise RuntimeError(f"build from_file failed! Error is {ret.Tostring()}")
  RuntimeError: build_from_file failed! Error is Common error code.
  [WARNING] ME(15411,7f07f56be100, python) : 2023-10-16-00:51:42.509.780 [mindspore/lite/src/extend rt/cxx_api/dlutils.h:124] DLSopen]
  dlopen /home/xxx/miniconda3/envs/yyy/lib/python3.8/site-packages/mindspore_lite/lib/libascend_kernel_plugin.so failed, error: libacl_cblas.so: cannot open shared object file: No such file or directory
  [ERROR] ME(15411,7f07f56be100, python) :2023-10-16-00:51:42.509.877 [mindspo re/lite/src/extendrt/kernel/ascend/plugin/ascend_allocator_plugin.cc:70] Register] DLSopen failed, so path: /home/xxx/miniconda3/envs/ yyy/lib/python3.8/site-packages/mindspore_lite/lib/libascend_kernel_plugin.so , func name: CreateAclAllocator. err: dlopen /home/xxx/miniconda3/envs/yyy/lib/python3.8/site-packages/mindspore_lite/lib/libascend_ kernel_plugin.so failed, error: libacl_cblas.so: cannot open shared object file: No such file or directory
  [ERROR] ME(15411,7f07f56be100, python):2023-10-16-00:51:42.509.893 [mindspore/lite/src/extendrt/infer_session.cc:66] HandleContext] failed register ascend allocator plugin.
  ...
  raise RuntimeError(f"build_from_file failed! Error is {ret.ToString()}")
  RuntimeError: build_from_file failed! Error is Common error code.
  ```

  This problem occur when `libascend_kernel plugin.so` is not successfully added to environment variables(`LD_LIBRARY_PATH`), which contained in `mindspore_lite` tar package. The solution is as follows:

  1. Check `mindspore_lite` tar package has been installed. If not, please follow the [MindSpore Lite download](https://www.mindspore.cn/lite/docs/en/master/use/downloads.html), install tar package and whl package. Please make sure the packages are Clound-side and support Ascend platform. See [MindSpore download](https://www.mindspore.cn/install/en) for details

  2. Find and move to the `mindspore_lite` installation path, like `/your_path_to/mindspore-lite`

  3. Run command `find ./ -name libascend_kernel_plugin.so` to find the so file, and you will get the following path:

  ```bash
  ./runtime/lib/libascend_kernel_plugin.so
  ```

  4. Add the path to to environment variables:

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

  This problem occur when `liboptiling.so` is not successfully added to environment variables(`LD_LIBRARY_PATH`). The solution is as follows:

  1. Check `CANN` tar package has been installed. If not, please follow the [Installing Ascend AI processor software package](https://www.mindspore.cn/install/en#installing-ascend-ai-processor-software-package) and install CANN

  2. Find and move to the `CANN` installation path, like `/your_path_to/cann`

  3. Run command `find ./ -name liboptiling.so` to find the so file, and you will get the following path:

     ```bash
     ./CANN-7.0/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64/liboptiling.so
     ./CANN-7.0/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/liboptiling.so
     ./CANN-7.0/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/minios/aarch64/liboptiling.so
     ```

  4. Add the path to to environment variables(e.g. x86_64 system):

     ```bash
     export LD_LIBRARY_PATH=$ASCEND_HOME/CANN-7.0/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/x86_64/:$LD_LIBRARY_PATH
     ```

     You will find the following message if the problem is solved:

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

  This problem occur when `libaicpu_ascend_engine.so` is not successfully added to environment variables(`LD_LIBRARY_PATH`). The solution is as follows:

  1. Check `CANN` tar package has been installed. If not, please follow the [Installing Ascend AI processor software package](https://www.mindspore.cn/install/en#installing-ascend-ai-processor-software-package) and install CANN
  2. Find and move to the `CANN` installation path, like `/your_path_to/cann`
  3. Run command `find ./ -name libaicpu_ascend_engine.so` to find the so file, and you will get the following path:

  ```bash
  ./CANN-7.0/x86_64-linux/lib64/plugin/opskernel/libaicpu_ascend_engine.so
  ./CANN-7.0/compiler/lib64/plugin/opskernel/libaicpu_ascend_engine.so
  ./latest/x86_64-linux/lib64/plugin/opskernel/libaicpu_ascend_engine.so
  ```

   4. Add the path to to environment variables(e.g. x86_64 system):

     ```bash
     export LD_LIBRARY_PATH=$ASCEND_HOME/CANN-7.0/compiler/lib64/plugin/opskernel/:$LD_LIBRARY_PATH
     ```


### Q3 Ascend Error Message A39999

- Error 1

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

- Error 2

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

Potential Reason:

- The mismatch of Ascend driver and CANN

- Unsuccessfully configuration of environment variables make aicpu fail, try to add the following items to environment variables(`LD_LIBRARY_PATH`):

  ```bash
  export ASCEND_OPP_PATH=${ASCEND_HOME}/latest/opp
  export ASCEND_AICPU_PATH=${ASCEND_OPP_PATH}/..
  ```

### Q4 `acl open device 0 failed`

Problem `acl open device 0 failed` may occur when doing inference. For example

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

This problem occur when `acllib` library is not successfully added to environment variables(`LD_LIBRARY_PATH`). try to add the following items to environment variables.

```bash
export NPU_HOST_LIB=$ASCEND_HOME/latest/acllib/lib64/stub
export DDK_PATH=$ASCEND_HOME/latest
export LD_LIBRARY_PATH=$ASCEND_HOME/latest/acllib/lib64
export ASCEND_AICPU_PATH=$ASCEND_HOME/latest/x86_64-linux
export LD_LIBRARY_PATH=$ASCEND_HOME/latest/x86_64-linux/lib64:$LD_LIBRARY_PATH
```

### Q5 Fail to install mindocr dependency on windows

Running the following command on windows

```bash
git clone git@gitee.com:mindspore-lab/mindocr.git
cd mindocr
pip install -e .
```

The error may occur with the following message, `lanms` package fail to install.

```bash
FileNotFoundError: [WinError 2]  the system cannot find the file specified.
```

`lanma` from [https://github.com/argman/EAST/](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2Fargman%2FEAST%2F)  on windows may not work, and linux platform is recommanded. You can try to use `lanms-neo` to replace `lanms` on windows. You may find the following error when installing `lanms-neo`:

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

The [Visual-cpp-build-tools](https://visualstudio.microsoft.com/zh-hans/visual-cpp-build-tools/) should be installed first, and you can successfully run `pip install lanms-neo`.

Remove the `lanms` item from `requirements.txt`, and run `pip install -r requirements.txt` to finish installation.

### Q6 `RuntimeError: The device address type is wrong: type name in address:CPU, type name in context:Ascend`

- `RuntimeError: The device address type is wrong: type name in address:CPU, type name in context:Ascend` may occur when exporting models:

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

- An error occurred in the calculation in Ascend mode, which raise `RuntimeError: The device address type is wrong: type name in address:CPU, type name in context:Ascend`

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
    File "/root/miniconda3/envs/py37/lib/python3.7/site-packages/mindspore/common/_stub_tensor.py", line 49, in fun
      return method(*arg, **kwargs)
    File "/root/miniconda3/envs/py37/lib/python3.7/site-packages/mindspore/common/tensor.py", line 493, in __str__
      return str(self.asnumpy())
    File "/root/miniconda3/envs/py37/lib/python3.7/site-packages/mindspore/common/tensor.py", line 964, in asnumpy
      return Tensor_.asnumpy(self)
  RuntimeError: The device address type is wrong: type name in address:CPU, type name in context:Ascend

  ----------------------------------------------------
  - C++ Call Stack: (For framework developers)
  ----------------------------------------------------
  mindspore/ccsrc/plugin/device/ascend/hal/hardware/ge_device_res_manager.cc:72 AllocateMemory
  ```

Ascend 310 or Ascend 310P3 may not support calculation in MindSpore Ascend mode. Please refer  [MindSpore download](https://www.mindspore.cn/install/en) and check the consistence of MindSpore version and platform. You can also

- Change Ascend mode to CPU
- Use [MindSpore Lite](https://www.mindspore.cn/lite/en) instead


### Q7 Problems related to model converting

- `SetGraphInputShape] Failed to find input xxx in input_shape yyy:xxxxxxxxxxx` occurs when converting model to Device-side `mindir` by `converter_lite`. If you convert [dbnet_resnet50.mindir](https://gitee.com/link?target=https%3A%2F%2Fdownload.mindspore.cn%2Ftoolkits%2Fmindocr%2Fdbnet%2Fdbnet_resnet50-c3a4aa24-fbf95c82.mindir) to Device-side `mindir` with the `config.txt` as following:

  ```bash
  [ascend_context]
  input_format=NCHW
  input_shape=args0:[1,3,736,1280]
  ```

  and run the following command to convert model:

  ```bash
  converter_lite --saveType=MINDIR --fmk=MINDIR --optimize=ascend_oriented --modelFile=dbnet_resnet50.mindir --outputFile=dbnet_resnet50_lite --configFile=config.txt
  ```

  Error may occur:

  ```bash
  [ERROR] LITE(30860,7f579d3f4f40,converter_lite):2023-11-10-03:19:29.005.385 [mindspore/lite/tools/converter/adapter/acl/src/acl_pass_impl.cc:756] SetGraphInputShape] Failed to find input x in input_shape args0:1,3,736,1280
  [ERROR] LITE(30860,7f579d3f4f40,converter_lite):2023-11-10-03:19:29.005.416 [mindspore/lite/tools/converter/adapter/acl/src/acl_pass_impl.cc:773] ConvertGraphToOm] Failed to set graph input shape
  [ERROR] LITE(30860,7f579d3f4f40,converter_lite):2023-11-10-03:19:29.005.427 [mindspore/lite/tools/converter/adapter/acl/src/acl_pass_impl.cc:862] BuildGraph] Convert graph  to om failed.
  [ERROR] LITE(30860,7f579d3f4f40,converter_lite):2023-11-10-03:19:29.005.439 [mindspore/lite/tools/converter/adapter/acl/src/acl_pass_impl.cc:1320] Run] Build graph failed.
  [ERROR] LITE(30860,7f579d3f4f40,converter_lite):2023-11-10-03:19:29.005.450 [mindspore/lite/tools/converter/adapter/acl/acl_pass.cc:42] Run] Acl pass impl run failed.
  [ERROR] LITE(30860,7f579d3f4f40,converter_lite):2023-11-10-03:19:29.005.461 [mindspore/lite/tools/converter/anf_transform.cc:472] RunConvertPass] Acl pass failed.
  [ERROR] LITE(30860,7f579d3f4f40,converter_lite):2023-11-10-03:19:29.005.476 [mindspore/lite/tools/converter/anf_transform.cc:660] RunPass] Run convert pass failed.
  [ERROR] LITE(30860,7f579d3f4f40,converter_lite):2023-11-10-03:19:29.005.486 [mindspore/lite/tools/converter/anf_transform.cc:754] TransformFuncGraph] Proc online transform failed.
  [ERROR] LITE(30860,7f579d3f4f40,converter_lite):2023-11-10-03:19:29.005.555 [mindspore/lite/tools/converter/anf_transform.cc:855] Transform] optimizer failed.
  [ERROR] LITE(30860,7f579d3f4f40,converter_lite):2023-11-10-03:19:29.005.564 [mindspore/lite/tools/converter/converter_funcgraph.cc:471] Optimize] Transform anf graph failed.
  [ERROR] LITE(30860,7f579d3f4f40,converter_lite):2023-11-10-03:19:29.006.118 [mindspore/lite/tools/converter/converter.cc:1029] HandleGraphCommon] Optimize func graph failed: -2 NULL pointer returned.
  [ERROR] LITE(30860,7f579d3f4f40,converter_lite):2023-11-10-03:19:29.013.133 [mindspore/lite/tools/converter/converter.cc:979] Convert] Handle graph failed: -2 NULL pointer returned.
  [ERROR] LITE(30860,7f579d3f4f40,converter_lite):2023-11-10-03:19:29.013.150 [mindspore/lite/tools/converter/converter.cc:1166] RunConverter] Convert model failed
  [ERROR] LITE(30860,7f579d3f4f40,converter_lite):2023-11-10-03:19:29.013.163 [mindspore/lite/tools/converter/cxx_api/converter.cc:348] Convert] Convert model failed, ret=NULL pointer returned.
  ERROR [mindspore/lite/tools/converter/converter_lite/main.cc:104] main] Convert failed. Ret: NULL pointer returned.
  Convert failed. Ret: NULL pointer returned.
  ```

  The problem is caused by the mismatch of variable names. The error message

  ```bash
  Failed to find input x in input_shape args0:1,3,736,1280
  ```

  shows that variable `x` does not match with variable `args0`. Try to replace `args0` with `x` in the `config.txt`.

- `Can't find OpAdapter for LSTM` occurs when converting model to `MindSpore Lite Mindir` by `converter_lite`.

  After exporting a MindIR file by `export.py` on lite inference devices, convert the MindIR file to `MindSpore Lite Mindir` by `converter_lite` as the following command:

  ```bash
  converter_lite \
    --saveType=MINDIR \
    --fmk=MINDIR \
    --optimize=ascend_oriented \
    --modelFile=./models/rec/CRNN/VGG7/crnn_vgg7.mindir \
    --outputFile=./models/rec/CRNN/VGG7/crnn_vgg7_lite \
    --configFile=./config.txt
  ```

  Error may occur:

  ```planetext
  [WARNING] GE_ADPT(837950,7feb6e13bf40,converter_lite):2024-10-26-07:37:40.545.361 [mindspore/ccsrc/transform/graph_ir/utils.cc:59] FindAdapter] Can't find OpAdapter for LSTM
  [ERROR] GE_ADPT(837950,7feb6e13bf40,converter_lite):2024-10-26-07:37:40.545.393 [mindspore/ccsrc/transform/graph_ir/convert.cc:4040] ConvertCNode] Cannot get adapter for Default/neck-RNNEncoder/seq_encoder-LSTM/rnn-_DynamicLSTMCPUGPU/LSTM-op90
  [ERROR] GE_ADPT(837950,7feb6e13bf40,converter_lite):2024-10-26-07:37:40.545.437 [mindspore/ccsrc/transform/graph_ir/convert.cc:1034] ConvertAllNode] Failed to convert node: @391_390_1_mindocr_models_base_model_BaseModel_construct_24_1:nout{[0]: ValueNode<Primitive> LSTM, [1]: nout, [2]: nout, [3]: nout, [4]: nout}.
  [ERROR] GE_ADPT(837950,7feb6e13bf40,converter_lite):2024-10-26-07:37:40.545.457 [mindspore/ccsrc/transform/graph_ir/convert.cc:1034] ConvertAllNode] Failed to convert node: ValueNode<Primitive> TupleGetItem.
  [ERROR] GE_ADPT(837950,7feb6e13bf40,converter_lite):2024-10-26-07:37:40.545.561 [mindspore/ccsrc/transform/graph_ir/convert.cc:1034] ConvertAllNode] Failed to convert node: @391_390_1_mindocr_models_base_model_BaseModel_construct_24_1:nout{[0]: ValueNode<Primitive> TupleGetItem, [1]: nout, [2]: ValueNode<Int64Imm> 0}.
  [ERROR] GE_ADPT(837950,7feb6e13bf40,converter_lite):2024-10-26-07:37:40.545.582 [mindspore/ccsrc/transform/graph_ir/convert.cc:1034] ConvertAllNode] Failed to convert node: ValueNode<Primitive> ReverseV2.
  [ERROR] GE_ADPT(837950,7feb6e13bf40,converter_lite):2024-10-26-07:37:40.545.667 [mindspore/ccsrc/transform/graph_ir/convert.cc:1034] ConvertAllNode] Failed to convert node: @391_390_1_mindocr_models_base_model_BaseModel_construct_24_1:nout{[0]: ValueNode<Primitive> ReverseV2, [1]: nout}.
  [ERROR] GE_ADPT(837950,7feb6e13bf40,converter_lite):2024-10-26-07:37:40.545.734 [mindspore/ccsrc/transform/graph_ir/convert.cc:1034] ConvertAllNode] Failed to convert node: @391_390_1_mindocr_models_base_model_BaseModel_construct_24_1:param_neck.seq_encoder.bias_hh_l0.
  ```

  In this case, please try to export the MindIR file by `export.py` on Ascend training devices, and then convert the MindIR file to `MindSpore Lite Mindir` by `converter_lite` on lite inference devices.

- `RuntimeError: Load op info form json config failed, version: Ascend310` occurs when export a MindIR file by `export.py`.

  Export a MindIR file by `export.py` on lite inference devices as the following command:

  ```bash
  python tools/export.py \
        --model_name_or_config configs/det/fcenet/fce_icdar15.yaml \
        --data_shape 736 1280 \
        --local_ckpt_path ./fcenet_resnet50-43857f7f.ckpt
  ```

  Error may occur:

  ```bash
    [ERROR] KERNEL(849474,7f7571d68740,python):2024-10-26-08:44:27.998.221 [mindspore/ccsrc/kernel/oplib/op_info_utils.cc:179] LoadOpInfoJson] Get op info json suffix path failed, soc_version: Ascend310
    [ERROR] KERNEL(849474,7f7571d68740,python):2024-10-26-08:44:27.998.362 [mindspore/ccsrc/kernel/oplib/op_info_utils.cc:118] GenerateOpInfos] Load op info json failed, version: Ascend310
    [ERROR] ANALYZER(849474,7f7571d68740,python):2024-10-26-08:44:30.168.028 [mindspore/ccsrc/pipeline/jit/ps/static_analysis/async_eval_result.cc:70] HandleException] Exception happened, check the information as below.
    RuntimeError: Load op info form json config failed, version: Ascend310

    ----------------------------------------------------
    - C++ Call Stack: (For framework developers)
    ----------------------------------------------------
    mindspore/ccsrc/plugin/device/ascend/hal/device/ascend_kernel_runtime.cc:320 Init
  ```

  In this case, please try to export the MindIR file by `export.py` on Ascend training devices.

- `Save ge model to buffer failed.` when using Cloud-Side `mindir` model to do inference.
  For example, if you use Cloud-Side `mindir` model to do inference in detection stage, it may raise:

  ```bash
  [ERROR] ME(43138,7f02bddd9740,python3):2023-11-10-03:40:45.206.120 [mindspore/ccsrc/cxx_api/model/acl/model_converter.cc:200] operator()] Save ge model to buffer failed.
  [ERROR] ME(43138,7f02bddd9740,python3):2023-11-10-03:40:45.206.157 [mindspore/ccsrc/cxx_api/model/model_converter_utils/multi_process.cc:118] ParentProcess] Parent process process failed
  [ERROR] ME(43123,7f02bddd9740,python3):2023-11-10-03:40:45.277.253 [mindspore/ccsrc/cxx_api/model/acl/model_converter.cc:200] operator()] Save ge model to buffer failed.
  [ERROR] ME(43123,7f02bddd9740,python3):2023-11-10-03:40:45.277.292 [mindspore/ccsrc/cxx_api/model/model_converter_utils/multi_process.cc:118] ParentProcess] Parent process process failed
  [ERROR] ME(43138,7f02bddd9740,python3):2023-11-10-03:40:46.235.224 [mindspore/ccsrc/cxx_api/model/acl/model_converter.cc:251] LoadMindIR] Convert MindIR model to OM model failed
  [ERROR] LITE(43138,7f02bddd9740,python3):2023-11-10-03:40:46.235.280 [mindspore/lite/tools/converter/adapter/acl/src/acl_pass_impl.cc:781] ConvertGraphToOm] Model converter load mindir failed.
  [ERROR] LITE(43138,7f02bddd9740,python3):2023-11-10-03:40:46.235.307 [mindspore/lite/tools/converter/adapter/acl/src/acl_pass_impl.cc:862] BuildGraph] Convert graph  to om failed.
  [ERROR] LITE(43138,7f02bddd9740,python3):2023-11-10-03:40:46.235.332 [mindspore/lite/tools/converter/adapter/acl/src/acl_pass_impl.cc:1320] Run] Build graph failed.
  [ERROR] LITE(43138,7f02bddd9740,python3):2023-11-10-03:40:46.235.359 [mindspore/lite/tools/converter/adapter/acl/acl_pass.cc:42] Run] Acl pass impl run failed.
  [ERROR] LITE(43138,7f02bddd9740,python3):2023-11-10-03:40:46.235.388 [mindspore/lite/tools/converter/anf_transform.cc:472] RunConvertPass] Acl pass failed.
  [ERROR] LITE(43138,7f02bddd9740,python3):2023-11-10-03:40:46.235.430 [mindspore/lite/tools/converter/anf_transform.cc:660] RunPass] Run convert pass failed.
  [ERROR] LITE(43138,7f02bddd9740,python3):2023-11-10-03:40:46.235.453 [mindspore/lite/tools/converter/anf_transform.cc:754] TransformFuncGraph] Proc online transform failed.
  [ERROR] LITE(43138,7f02bddd9740,python3):2023-11-10-03:40:46.235.673 [mindspore/lite/tools/converter/anf_transform.cc:855] Transform] optimizer failed.
  [ERROR] LITE(43138,7f02bddd9740,python3):2023-11-10-03:40:46.235.698 [mindspore/lite/tools/converter/converter_funcgraph.cc:471] Optimize] Transform anf graph failed.
  [ERROR] ME(43138,7f02bddd9740,python3):2023-11-10-03:40:46.238.216 [mindspore/lite/src/extendrt/convert/runtime_convert.cc:214] RuntimeConvert] Convert model failed
  [ERROR] ME(43138,7f02bddd9740,python3):2023-11-10-03:40:46.238.270 [mindspore/lite/src/extendrt/cxx_api/model/model_impl.cc:507] ConvertGraphOnline] Failed to converter graph
  [ERROR] ME(43138,7f02bddd9740,python3):2023-11-10-03:40:46.238.351 [mindspore/lite/src/extendrt/cxx_api/model/model_impl.cc:395] BuildByBufferImpl] convert graph failed.
  [ERROR] MINDOCR(43138:139649752078144,Process-1:18):2023-11-10-03:40:46.255.926 [src/parallel/framework/module_base.py:38] DetInferNode init failed: build_from_file failed! Error is Common error code.
  Process Process-1:18:
  Traceback (most recent call last):
    File "/root/miniconda3/envs/py37/lib/python3.7/multiprocessing/process.py", line 297, in _bootstrap
      self.run()
    File "/root/miniconda3/envs/py37/lib/python3.7/multiprocessing/process.py", line 99, in run
      self._target(*self._args, **self._kwargs)
    File "/home/mindocr/deploy/py_infer/src/parallel/framework/module_base.py", line 39, in process_handler
      raise error
    File "/home/mindocr/deploy/py_infer/src/parallel/framework/module_base.py", line 34, in process_handler
      params = self.init_self_args()
    File "/home/mindocr/deploy/py_infer/src/parallel/module/detection/det_infer_node.py", line 12, in init_self_args
      self.text_detector.init(preprocess=False, model=True, postprocess=False)
    File "/home/mindocr/deploy/py_infer/src/infer/infer_base.py", line 29, in init
      self._init_model()
    File "/home/mindocr/deploy/py_infer/src/infer/infer_det.py", line 22, in _init_model
      device_id=self.args.device_id,
    File "/home/mindocr/deploy/py_infer/src/core/model/model.py", line 15, in __init__
      self.model = _INFER_BACKEND_MAP[backend](**kwargs)
    File "/home/mindocr/deploy/py_infer/src/core/model/backend/lite_model.py", line 16, in __init__
      super().__init__(model_path, device, device_id)
    File "/home/mindocr/deploy/py_infer/src/core/model/backend/model_base.py", line 28, in __init__
      self._init_model()
    File "/home/mindocr/deploy/py_infer/src/core/model/backend/lite_model.py", line 33, in _init_model
      self.model.build_from_file(self.model_path, mslite.ModelType.MINDIR, context)
    File "/root/miniconda3/envs/py37/lib/python3.7/site-packages/mindspore_lite/model.py", line 95, in warpper
      return func(*args, **kwargs)
    File "/root/miniconda3/envs/py37/lib/python3.7/site-packages/mindspore_lite/model.py", line 235, in build_from_file
      raise RuntimeError(f"build_from_file failed! Error is {ret.ToString()}")
  RuntimeError: build_from_file failed! Error is Common error code.
  [ERROR] ME(43123,7f02bddd9740,python3):2023-11-10-03:40:46.305.698 [mindspore/ccsrc/cxx_api/model/acl/model_converter.cc:251] LoadMindIR] Convert MindIR model to OM model failed
  [ERROR] LITE(43123,7f02bddd9740,python3):2023-11-10-03:40:46.305.755 [mindspore/lite/tools/converter/adapter/acl/src/acl_pass_impl.cc:781] ConvertGraphToOm] Model converter load mindir failed.
  [ERROR] LITE(43123,7f02bddd9740,python3):2023-11-10-03:40:46.305.782 [mindspore/lite/tools/converter/adapter/acl/src/acl_pass_impl.cc:862] BuildGraph] Convert graph  to om failed.
  [ERROR] LITE(43123,7f02bddd9740,python3):2023-11-10-03:40:46.305.807 [mindspore/lite/tools/converter/adapter/acl/src/acl_pass_impl.cc:1320] Run] Build graph failed.
  [ERROR] LITE(43123,7f02bddd9740,python3):2023-11-10-03:40:46.305.834 [mindspore/lite/tools/converter/adapter/acl/acl_pass.cc:42] Run] Acl pass impl run failed.
  [ERROR] LITE(43123,7f02bddd9740,python3):2023-11-10-03:40:46.305.864 [mindspore/lite/tools/converter/anf_transform.cc:472] RunConvertPass] Acl pass failed.
  [ERROR] LITE(43123,7f02bddd9740,python3):2023-11-10-03:40:46.305.904 [mindspore/lite/tools/converter/anf_transform.cc:660] RunPass] Run convert pass failed.
  [ERROR] LITE(43123,7f02bddd9740,python3):2023-11-10-03:40:46.305.928 [mindspore/lite/tools/converter/anf_transform.cc:754] TransformFuncGraph] Proc online transform failed.
  [ERROR] LITE(43123,7f02bddd9740,python3):2023-11-10-03:40:46.306.162 [mindspore/lite/tools/converter/anf_transform.cc:855] Transform] optimizer failed.
  [ERROR] LITE(43123,7f02bddd9740,python3):2023-11-10-03:40:46.306.188 [mindspore/lite/tools/converter/converter_funcgraph.cc:471] Optimize] Transform anf graph failed.
  [ERROR] ME(43123,7f02bddd9740,python3):2023-11-10-03:40:46.308.599 [mindspore/lite/src/extendrt/convert/runtime_convert.cc:214] RuntimeConvert] Convert model failed
  [ERROR] ME(43123,7f02bddd9740,python3):2023-11-10-03:40:46.308.646 [mindspore/lite/src/extendrt/cxx_api/model/model_impl.cc:507] ConvertGraphOnline] Failed to converter graph
  [ERROR] ME(43123,7f02bddd9740,python3):2023-11-10-03:40:46.308.712 [mindspore/lite/src/extendrt/cxx_api/model/model_impl.cc:395] BuildByBufferImpl] convert graph failed.
  [ERROR] MINDOCR(43123:139649752078144,Process-1:17):2023-11-10-03:40:46.324.506 [src/parallel/framework/module_base.py:38] DetPreNode init failed: build_from_file failed! Error is Common error code.
  Process Process-1:17:
  Traceback (most recent call last):
    File "/root/miniconda3/envs/py37/lib/python3.7/multiprocessing/process.py", line 297, in _bootstrap
      self.run()
    File "/root/miniconda3/envs/py37/lib/python3.7/multiprocessing/process.py", line 99, in run
      self._target(*self._args, **self._kwargs)
    File "/home/mindocr/deploy/py_infer/src/parallel/framework/module_base.py", line 39, in process_handler
      raise error
    File "/home/mindocr/deploy/py_infer/src/parallel/framework/module_base.py", line 34, in process_handler
      params = self.init_self_args()
    File "/home/mindocr/deploy/py_infer/src/parallel/module/detection/det_pre_node.py", line 13, in init_self_args
      self.text_detector.init(preprocess=True, model=False, postprocess=False)
    File "/home/mindocr/deploy/py_infer/src/infer/infer_base.py", line 29, in init
      self._init_model()
    File "/home/mindocr/deploy/py_infer/src/infer/infer_det.py", line 22, in _init_model
      device_id=self.args.device_id,
    File "/home/mindocr/deploy/py_infer/src/core/model/model.py", line 15, in __init__
      self.model = _INFER_BACKEND_MAP[backend](**kwargs)
    File "/home/mindocr/deploy/py_infer/src/core/model/backend/lite_model.py", line 16, in __init__
      super().__init__(model_path, device, device_id)
    File "/home/mindocr/deploy/py_infer/src/core/model/backend/model_base.py", line 28, in __init__
      self._init_model()
    File "/home/mindocr/deploy/py_infer/src/core/model/backend/lite_model.py", line 33, in _init_model
      self.model.build_from_file(self.model_path, mslite.ModelType.MINDIR, context)
    File "/root/miniconda3/envs/py37/lib/python3.7/site-packages/mindspore_lite/model.py", line 95, in warpper
      return func(*args, **kwargs)
    File "/root/miniconda3/envs/py37/lib/python3.7/site-packages/mindspore_lite/model.py", line 235, in build_from_file
      raise RuntimeError(f"build_from_file failed! Error is {ret.ToString()}")
  RuntimeError: build_from_file failed! Error is Common error code.
  ```

Reason:
- Cloud-Side `mindir` model should convert to Device-Side `mindir` model first.
- Mismatch version of `converter_lite` tool and `mindspore_lite`. For example, it may fail when using `converter_lite 2.2` to get Device-Side `mindir`, and do inference with `mindspore_lite 2.1`.

### Q8 Problems related to inference

- When doing inference with `deploy/py_infer/infer.py`, it may occur `TypeError: unhashable type: 'numpy.ndarray'`

  ```bash
  [ERROR] MINDOCR(51913:140354674829120,Process-1:28):2023-11-10-06:52:34.304.673 [src/parallel/framework/module_base.py:66] ERROR occurred in RecPostNode module for test.jpg: unhashable type: 'numpy.ndarray'.
  Traceback (most recent call last):
    File "/home/mindocr/deploy/py_infer/src/parallel/framework/module_base.py", line 62, in call_process
      self.process(send_data)
    File "/home/mindocr/deploy/py_infer/src/parallel/module/recognition/rec_post_node.py", line 24, in process
      output = self.text_recognizer.postprocess(data["pred"], batch)
    File "/home/mindocr/deploy/py_infer/src/infer/infer_rec.py", line 132, in postprocess
      return self.postprocess_ops(pred)
    File "/home/mindocr/deploy/py_infer/src/data_process/postprocess/builder.py", line 32, in __call__
      return self._ops_func(*args, **kwargs)
    File "/home/mindocr/mindocr/postprocess/rec_postprocess.py", line 153, in __call__
      raw_chars = [[self.character[idx] for idx in pred_indices[b]] for b in range(pred_indices.shape[0])]
    File "/home/mindocr/mindocr/postprocess/rec_postprocess.py", line 153, in <listcomp>
      raw_chars = [[self.character[idx] for idx in pred_indices[b]] for b in range(pred_indices.shape[0])]
    File "/home/mindocr/mindocr/postprocess/rec_postprocess.py", line 153, in <listcomp>
      raw_chars = [[self.character[idx] for idx in pred_indices[b]] for b in range(pred_indices.shape[0])]
  TypeError: unhashable type: 'numpy.ndarray'
  ```

  This problem occurs due to shape error, please check:

  - Use suitable model. For example, it may fail and pass detection model to `--rec_model_path` parameter.
  - Use inference model(not training model) to do converting.

### Q9 Training speed of DBNet not as fast as expexted

When traning DBNet series networks (including DBNet MobileNetV3, DBNet ResNet-18, DBNet ResNet-50, and DBNet++ ResNet-50) using following command, the training speed is not as fast as expexted. For instance, the training speed of DBNet MobileNetV3 can reach only 80fps which is slower than the expecting 100fps.

``` bash
python tools/train.py -c configs/det/dbnet/db_mobilenetv3_icdar15.yaml
```

This problem is due to the complex data pre-processing procedures of DBNet. The data pre-processing procedures will become the performance bottleneck if the computation ability of a CPU core of the training server is relatively weak.

**Solutions**

1. Try to set the `train.dataset.use_minddata` and `eval.dataset.use_minddata` in the configuration file to `True`. MindOCR will execute parts of data pre-processing procedures using MindSpore[MindData](https://www.mindspore.cn/docs/zh-CN/master/api_python/dataset/dataset_method/operation/mindspore.dataset.Dataset.map.html?highlight=map#mindspore.dataset.Dataset.map):

    ``` yaml
    ...
    train:
      ckpt_save_dir: './tmp_det'
      dataset_sink_mode: True
      dataset:
        type: DetDataset
        dataset_root: /data/ocr_datasets
        data_dir: ic15/det/train/ch4_training_images
        label_file: ic15/det/train/det_gt.txt
        sample_ratio: 1.0
        use_minddata: True                          <-- Set this configuration
    ...
    eval:
      ckpt_load_path: tmp_det/best.ckpt
      dataset_sink_mode: False
      dataset:
        type: DetDataset
        dataset_root: /data/ocr_datasets
        data_dir: ic15/det/test/ch4_test_images
        label_file: ic15/det/test/det_gt.txt
        sample_ratio: 1.0
        use_minddata: True                          <-- Set this configuration
    ...
    ```

2. Try to set the `train.loader.num_workers` in the configuration file to a larger value to enhance the number of threads fetching dataset if the training server has enough CPU cores:

    ``` yaml
    ...
    train:
      ...
      loader:
        shuffle: True
        batch_size: 10
        drop_remainder: True
        num_workers: 12                             <-- Set this configuration
    ...
    ```

### Q10 Error about `libgomp-d22c30c5.so.1.0.0`
The following error may occur when running mindocr
```bash
ImportError: /root/mindocr_env/lib/python3.8/site-packages/sklearn/__check_build/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0: cannot allocate memory in static TLS block
```
You can try the following steps to fix it:
 - search `libgomp-d22c30c5.so.1.0.0` in your python install path
   ```bash
   cd /root/mindocr_env/lib/python3.8
   find ~ -name libgomp-d22c30c5.so.1.0.0
   ```
   and get the following search result:
   ```bash
   /root/mindocr_env/lib/python3.8/site-packages/scikit_learn.libs/libgomp-d22c30c5.so.1.0.0
   ```
 - Add the so file to environment variable `LD_PRELOAD`
   ```bash
   export LD_PRELOAD=/root/mindocr_env/lib/python3.8/site-packages/scikit_learn.libs/libgomp-d22c30c5.so.1.0.0:$LD_PRELOAD
   ```

### Q11 Dataset Pipeline Error when training abinet on lmdb dataset
The following error may occur when training abinet on lmdb dataset
```bash
mindocr.data.rec_lmdb_dataset WARNING - Error occurred during preprocess.
 Exception thrown from dataset pipeline. Refer to 'Dataset Pipeline Error Message'.

------------------------------------------------------------------
- Dataset Pipeline Error Message:
------------------------------------------------------------------
[ERROR] No cast for the specified DataType was found.

------------------------------------------------------------------
- C++ Call Stack: (For framework developers)
------------------------------------------------------------------
mindspore/ccsrc/minddata/dataset/kernels/py_func_op.cc(143).
```
You can try the following steps to fix it:
 - find the folder of mindspore package
 - open file: `mindspore/dataset/transforms/transform.py`
 - switch to line 93:
  ```bash
  93        if key in EXECUTORS_LIST:
  94           # get the executor by process id and thread id
  95            executor = EXECUTORS_LIST[key]
  96            # remove the old transform which in executor and update the new transform
  97            executor.UpdateOperation(self.parse())
  98        else:
  99            # create a new executor by process id and thread_id
  100           executor = cde.Execute(self.parse())
  101           # add the executor the global EXECUTORS_LIST
  102           EXECUTORS_LIST[key] = executor
  ```
 - replace line 97 with `executor = cde.Execute(self.parse())`, and get
  ```bash
  93        if key in EXECUTORS_LIST:
  94            # get the executor by process id and thread id
  95            executor = EXECUTORS_LIST[key]
  96            # remove the old transform which in executor and update the new transform
  97            executor = cde.Execute(self.parse())
  98        else:
  99            # create a new executor by process id and thread_id
  100           executor = cde.Execute(self.parse())
  101           # add the executor the global EXECUTORS_LIST
  102           EXECUTORS_LIST[key] = executor
  ```
  - save the file, and try to train the model.


### Q12 Runtime Error when training dbnet on synthtext dataset
Runtime Error occur as following when training dbnet on synthtext dataset:
```bash
Traceback (most recent call last):
  ...
  File "/root/archiconda3/envs/Python380/lib/python3.8/site-packages/mindspore/common/api.py", line 1608, in _exec_pip
    return self.graph_executor(args, phase)
RuntimeError: Run task for graph:kernel_graph_1 error! The details reger to 'Ascend Error Message'
```
Please update CANN to 7.1 version.


### Q13 Failed to install seqeval
The following error occur when run `pip install -r requirements.txt`
```bash
Collecting seqeval>=1.2.2 (from -r requirements.txt (line 19))
  Downloading http://mirrors.aliyun.com/pypi/packages/9d/2d/233c79d5b4e5ab1dbf111242299153f3caddddbb691219f363ad55ce783d/seqeval-1.2.2.tar.gz (43 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 43.6/43.6 kB 181.0 kB/s eta 0:00:00
  Preparing metadata (setup.py) ... error
  error: subprocess-exited-with-error

  × python setup.py egg_info did not run successfully.
  │ exit code: 1
  ╰─> [48 lines of output]
      /home/ma-user/anaconda3/envs/MindSpore/lib/python3.9/site-packages/setuptools/__init__.py:80: _DeprecatedInstaller: setuptools.installer and fetch_build_eggs are deprecated.
      !!

              ********************************************************************************
              Requirements should be satisfied by a PEP 517 installer.
              If you are using pip, you can try `pip install --use-pep517`.
              ********************************************************************************

      !!
        dist.fetch_build_eggs(dist.setup_requires)
      WARNING: The repository located at mirrors.aliyun.com is not a trusted or secure host and is being ignored. If this repository is available via HTTPS we recommend you use HTTPS instead, otherwise you may silence this warning and allow it anyway with '--trusted-host mirrors.aliyun.com'.
      ERROR: Could not find a version that satisfies the requirement setuptools_scm (from versions: none)
      ERROR: No matching distribution found for setuptools_scm
      Traceback (most recent call last):
        File "/home/ma-user/anaconda3/envs/MindSpore/lib/python3.9/site-packages/setuptools/installer.py", line 101, in _fetch_build_egg_no_warn
          subprocess.check_call(cmd)
        File "/home/ma-user/anaconda3/envs/MindSpore/lib/python3.9/subprocess.py", line 373, in check_call
          raise CalledProcessError(retcode, cmd)
      subprocess.CalledProcessError: Command '['/home/ma-user/anaconda3/envs/MindSpore/bin/python3.9', '-m', 'pip', '--disable-pip-version-check', 'wheel', '--no-deps', '-w', '/tmp/tmpusgt0k69', '--quiet', 'setuptools_scm']' returned non-zero exit status 1.

      The above exception was the direct cause of the following exception:

      Traceback (most recent call last):
        File "<string>", line 2, in <module>
        File "<pip-setuptools-caller>", line 34, in <module>
        File "/tmp/pip-install-m2kqztlz/seqeval_da00f708dc0e483b92cd18083513d5e7/setup.py", line 27, in <module>
          setup(
        File "/home/ma-user/anaconda3/envs/MindSpore/lib/python3.9/site-packages/setuptools/__init__.py", line 102, in setup
          _install_setup_requires(attrs)
        File "/home/ma-user/anaconda3/envs/MindSpore/lib/python3.9/site-packages/setuptools/__init__.py", line 75, in _install_setup_requires
          _fetch_build_eggs(dist)
        File "/home/ma-user/anaconda3/envs/MindSpore/lib/python3.9/site-packages/setuptools/__init__.py", line 80, in _fetch_build_eggs
          dist.fetch_build_eggs(dist.setup_requires)
        File "/home/ma-user/anaconda3/envs/MindSpore/lib/python3.9/site-packages/setuptools/dist.py", line 636, in fetch_build_eggs
          return _fetch_build_eggs(self, requires)
        File "/home/ma-user/anaconda3/envs/MindSpore/lib/python3.9/site-packages/setuptools/installer.py", line 38, in _fetch_build_eggs
          resolved_dists = pkg_resources.working_set.resolve(
        File "/home/ma-user/anaconda3/envs/MindSpore/lib/python3.9/site-packages/pkg_resources/__init__.py", line 829, in resolve
          dist = self._resolve_dist(
        File "/home/ma-user/anaconda3/envs/MindSpore/lib/python3.9/site-packages/pkg_resources/__init__.py", line 865, in _resolve_dist
          dist = best[req.key] = env.best_match(
        File "/home/ma-user/anaconda3/envs/MindSpore/lib/python3.9/site-packages/pkg_resources/__init__.py", line 1135, in best_match
          return self.obtain(req, installer)
        File "/home/ma-user/anaconda3/envs/MindSpore/lib/python3.9/site-packages/pkg_resources/__init__.py", line 1147, in obtain
          return installer(requirement)
        File "/home/ma-user/anaconda3/envs/MindSpore/lib/python3.9/site-packages/setuptools/installer.py", line 103, in _fetch_build_egg_no_warn
          raise DistutilsError(str(e)) from e
      distutils.errors.DistutilsError: Command '['/home/ma-user/anaconda3/envs/MindSpore/bin/python3.9', '-m', 'pip', '--disable-pip-version-check', 'wheel', '--no-deps', '-w', '/tmp/tmpusgt0k69', '--quiet', 'setuptools_scm']' returned non-zero exit status 1.
      [end of output]

  note: This error originates from a subprocess, and is likely not a problem with pip.
error: metadata-generation-failed

× Encountered error while generating package metadata.
╰─> See above for output.

note: This is an issue with the package mentioned above, not pip.

```
Please try the following steps to fix this problem：
 - Update `setuptools`: `pip3 install --upgrade setuptools`
 - Update `setuptools_scm`: `pip3 install --upgrade setuptools_scm`
 - Install `seqeval`：`pip3 install seqeval -i https://pypi.tuna.tsinghua.edu.cn/simple`


### Q14 Failed to install lanms
The following error occur when installing lanms
```bash
ImportError: Python version mismatch: module was compiled for version 3.8, while the interpreter is running version 3.7.
```
Some Python 3.7 environment may meet this problem when multiple python3 environment exists. You could try the following steps to solve this problem:
1. run `pip3 install lanms -i https://pypi.tuna.tsinghua.edu.cn/simple`, and get the url for downloading `lanms-1.0.2.tar.gz`(like https://pypi.tuna.tsinghua.edu.cn/packages/96/c0/50dc2c857ed060e907adaef31184413a7706e475c322236d346382e45195/lanms-1.0.2.tar.gz)
2. use this url and dowload the `lanms-1.0.2.tar.gz`, run `tar -zxvf lanms-1.0.2.tar.gz` to decompress the package.
3. `cd lanms-1.0.2`
4. edit the `Makefile`, replace `python3-config` with `python3.7-config` in line 1 and line 2, and you could get
   ```text
   CXXFLAGS = -I include  -std=c++11 -O3 $(shell python3.7-config --cflags)
   LDFLAGS = $(shell python3.7-config --ldflags)
   ...
   ```
   save `Makefile`. So that the make process would exactly compile with python3.7 environment
5. run `python setup.py install` and completely install lanms.
>>>>>>> dca11cc9989deabe86985f0729502266e5ba6f42
