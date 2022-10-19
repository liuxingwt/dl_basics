## 1. 环境 + TensorRT OSS + TensorRT GA
+ 指定环境：Ubuntu18.04，Cuda10.2，Cudnn8.1，TensorRT7.1.3.4
+ 
```shell
（1）使用TensorRT OSS

（2.1）按照TensorRT github项目README.md中的Prerequisites配置：
环境 + TensorRT GA

（2.2）使用和TensorRT版本对应的TensorRT OSS分支
git clone -b 7.1.3 https://github.com/NVIDIA/TensorRT.git
cd TensorRT
git submodule update --init --recursive
export TRT_SOURCE=`pwd`
# 构建容器环境
./docker/build.sh --file docker/ubuntu.Dockerfile --tag tensorrt-ubuntu --os 18.04 --cuda 10.2
# 备注：这个版本的dockerfile中Download NGC client行有bug，可以用master分支dockerfile中对应行替换

```

#### 1.2 安装TensorRT
```shell

```
