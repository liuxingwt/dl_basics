### 1.下载项目
+ 环境：Ubuntu18.04，Cuda10.2，Cudnn8.1，TensorRT7.1.3.4
```shell
# 拉取TensorRT版本对应的TensorRT OSS分支
git clone -b 7.1.3 https://github.com/NVIDIA/TensorRT.git
```

### 2.构建环境
```shell
./docker/build.sh --file docker/ubuntu.Dockerfile --tag tensorrt-ubuntu --os 18.04 --cuda 10.2
```
