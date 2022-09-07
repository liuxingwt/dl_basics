#### 2. onnx模型转TensorRT
##### （1）安装TensorRT
+ Linux平台上，下载TensorRT OSS，项目github地址：https://github.com/NVIDIA/TensorRT 
```
$ git clone -b master https://github.com/nvidia/TensorRT TensorRT
$ cd TensorRT
$ git submodule update --init --recursive
$ export TRT_SOURCE=`pwd`
```
+ 依据TensorRT OSS搭建环境：依据NVIDIA/TensorRT项目中的docker/ubuntu.Dockerfile文件，在本地build镜像  
`$ ./docker/build.sh --file docker/ubuntu.Dockerfile --tag tensorrt-ubuntu --os 18.04 --cuda 11.0`  
然后在本地使用`docker run`构建容器；  
也可以用项目readme里的prerequisites自己手动安装依赖环境。  
+ 下载并安装TensorRT GA：依据ubuntu和cuda的版本下载对应的tensorRT版本。  
可以直接从内网机器上拉取，地址：http://10.43.26.179/nvidia/tensorrt/  
下载对应的TensorRT版本后，根据官网的安装说明 4.1Debian installation来安装tensorrt。  
https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-debian  
具体步骤如下：
```
# 安装TensorRT
$ sudo dpkg -i nv-tensorrt-repo-ubuntu1604-ga-cuda8.0-trt3.0-20171128_1-1_amd64.deb
$ sudo apt-get update
$ sudo apt-get install tensorrt

$ apt-get install python3-libnvinfer-dev     # Python3安装
$ sudo apt-get install uff-converter-tf      # tensorflow需要安装
$ dpkg -l | grep TensorRT                    # 验证安装结果
```

##### (2) onnx或caffe模型转TensorRT
```
# 找到trtexec所在目录，通常位于/usr/src/tensorrt/bin/trtexec
$ find /usr -name "*trtexec"

# 通过help查看trtexec的使用方法
$ /usr/src/tensorrt/bin/trtexec -help

# 将pfld.onnx模型转换为pfld.trt格式的模型
$ /usr/src/tensorrt/bin/trtexec --onnx=pfld.onnx --saveEngine=pfld.engine

# caffe模型转换为tensorrt
$ /usr/src/tensorrt/bin/trtexec --model=quality_mobilenet_v2.caffemodel --deploy=quality_mobilenet_v2.prototxt --output=liveness_prob --batch=1 --saveEngine=liveness_mnv2.trt
```

##### (3) TesorRT模型测试：
```shell
# 安装opencv和pycuda
$ apt-get install libsm6 libxext6 libxrender1
$ pip install -i https://mirrors.aliyun.com/pypi/simple/ opencv-python pycuda 

# 使用tensorRT框架下的推理引擎pfld.engine，计算example.jpg的landmark
# 参考资料：https://github.com/rmccorm4/tensorrt-utils/blob/20.03/classification/imagenet/infer_tensorrt_imagenet.py
# 这两段tensorrt的python测试代码的真正来源：https://github.com/NVIDIA/sampleQAT
$ python test_tensorrt.py --engine pfld.engine --file example.jpg
```