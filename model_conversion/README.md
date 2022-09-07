### 模型转换
#### 1. pytorch模型转为onnx
#### 2. onnx模型转TensorRT
#### 3. Pytorch模型转caffe
#### 4. caffe模型转ncnn

#### 5. 失败的转换 
##### 5.1 onnx模型转caffe失败经历
```shell
# 通过docker镜像安装caffe的cpu版本，使用gpu版跑demo时可能会报cudnn的错
docker pull bvlc/caffe:cpu
docker run -it -p 899:8999 -v $PWD:/home --name caffe_cpu_lx bvlc/caffe:cpu bash

# 安装onnx
pip install -i https://mirrors.aliyun.com/pypi/simple onnx

# 使用下面这个项目进行转换，先测试demo能否跑通
# 附caffe模型可视化：http://ethereon.github.io/netscope/quickstart.html
git clone https://github.com/MTlab/onnx2caffe.git
cd onnx2caffe
python convertCaffe.py ./model/MobileNetV2.onnx ./model/MobileNetV2.prototxt ./model/MobileNetV2.caffemodel

# 使用netron可视化onnx模型，网址：https://lutzroeder.github.io/netron/
# 直接将onnx模型转caffe报错如下：
TypeError: ONNX node of type Pad is not supported.
```

#### 6. 未做的转换
##### 6.1 onnx模型转openvino
（1）OpenVINO简介
+ OpenVINO全称是Open Visual Inference and Neural Network Optimization
+ 支持设备：Intel CPU、GPU、FPGA和Movidius计算棒等多种设备；
+ OpenVINO工具包的主要组件是DLDT(Deep Learning Deployment Toolkit，深度学习部署工具包)。
+ DLDT主要包括模型优化器(Model Optimizer)和推理引擎（Inference engine，IE）两部分
+ OpenVINO参考资料：https://zhuanlan.zhihu.com/p/129879495 
+ 使用OpenVINO部署深度学习应用：https://mp.weixin.qq.com/s/LSYgDK1RhhT90SrCEzGLiQ
+ 

##### 6.2 onnx模型转ncnn
+ 参考资料：https://github.com/Tencent/ncnn/wiki/how-to-build





