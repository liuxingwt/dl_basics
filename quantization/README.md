## TensorRT模型INT8量化

### 1. 构建：TensorRT OSS + 基础环境 + TensorRT GA
+ 指定环境：Ubuntu18.04，Cuda10.2，Cudnn8.1，TensorRT7.1.3.4
```shell
（1）下载TensorRT OSS
# 使用和TensorRT版本对应的TensorRT OSS分支
git clone -b 7.1.3 https://github.com/NVIDIA/TensorRT.git
cd TensorRT
git submodule update --init --recursive
export TRT_SOURCE=`pwd`

（2）下载并解压TensorRT GA：.tar包的方式）
# 其他方法：使用.deb包的方式参考https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-debian
# 下载安装地址：https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#downloading
# 依据ubuntu和cuda的版本下载对应的tensorRT版本
cd ~/Downloads
tar -xvzf TensorRT-7.1.3.4.Ubuntu-18.04.x86_64-gnu.cuda-10.2.cudnn8.0.tar.gz
export TRT_RELEASE=`pwd`/TensorRT-7.1.3.4

（3）创建并启动容器环境（也可以使用TensorRT OSS README中的Prerequisites）
# 使用和TensorRT版本对应的TensorRT OSS分支
./docker/build.sh --file docker/ubuntu.Dockerfile --tag tensorrt-ubuntu --os 18.04 --cuda 10.2
# 备注1：这个版本的dockerfile中Download NGC client行有bug，可以用master分支dockerfile中对应行替换
# 备注2：注释掉了dockerfile中 USER trtuser行
# 启动docker容器
./docker/launch.sh --tag tensorrt-ubuntu --gpus all --release $TRT_RELEASE --source $TRT_SOURCE

（4）构建整个TensorRT OSS系统
cd $TRT_SOURCE
mkdir -p build && cd build
cmake .. -DTRT_LIB_DIR=$TRT_RELEASE/lib -DTRT_OUT_DIR=`pwd`/out -DCUDA_VERSION=10.2   # 记得加上cuda版本
make -j$(nproc)

（5）验证trtexec安装是否成功
cd /workspace/TensorRT/downloads/TensorRT-7.1.3.4/bin/
./trtexec --help
# 备注：如果报错./trtexec: error while loading shared libraries: libnvcaffeparser.so.7: cannot open shared object file: No such file or directory
# 解决：export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/workspace/TensorRT/downloads/TensorRT-7.1.3.4/lib
```

### 2. 跑通TensorRT提供的sample
(1) 在python代码中使用tensorrt
```shell
# 参考：https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-tar
# Install the Python TensorRT wheel file
cd TensorRT-${version}/python
python3 -m pip install tensorrt-*-cp3x-none-linux_x86_64.whl
# Install the Python graphsurgeon wheel file.
cd TensorRT-${version}/graphsurgeon
python3 -m pip install graphsurgeon-0.4.6-py2.py3-none-any.whl
# Install the Python onnx-graphsurgeon wheel file.
cd TensorRT-${version}/onnx_graphsurgeon
python3 -m pip install onnx_graphsurgeon-0.3.12-py2.py3-none-any.whl
```





