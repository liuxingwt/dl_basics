## 1. TensorRT OSS + 环境 + TensorRT GA
+ 指定环境：Ubuntu18.04，Cuda10.2，Cudnn8.1，TensorRT7.1.3.4
```shell
（1）下载TensorRT OSS
# 使用和TensorRT版本对应的TensorRT OSS分支
git clone -b 7.1.3 https://github.com/NVIDIA/TensorRT.git
cd TensorRT
git submodule update --init --recursive
export TRT_SOURCE=`pwd`

（2）构建容器环境
# 使用和TensorRT版本对应的TensorRT OSS分支
./docker/build.sh --file docker/ubuntu.Dockerfile --tag tensorrt-ubuntu --os 18.04 --cuda 10.2
# 备注：这个版本的dockerfile中Download NGC client行有bug，可以用master分支dockerfile中对应行替换

（3）安装TensorRT GA
# 依据ubuntu和cuda的版本下载对应的tensorRT版本
# 下载安装地址：https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#downloading
# 下载对应的TensorRT版本后，根据官网的安装说明 4.1Debian installation来安装tensorrt
sudo dpkg -i nv-tensorrt-repo-ubuntu1604-ga-cuda8.0-trt3.0-20171128_1-1_amd64.deb
sudo apt-get update
sudo apt-get install tensorrt

apt-get install python3-libnvinfer-dev     # Python3安装
sudo apt-get install uff-converter-tf      # tensorflow需要安装
dpkg -l | grep TensorRT                    # 验证安装结果
```

## 2. 使用TensorRT

