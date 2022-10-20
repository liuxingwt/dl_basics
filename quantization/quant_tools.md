## 量化工具介绍
### 1.量化基础概念
（1）两种量化方式
+ **QAT**：是Quantization Aware Training的简称；  
+ **PTQ**：是Post Training Quantization的简称。


### 2.使用TensorRT工具原生的INT8量化
##### 2.1 PTQ方式
+ NVIDIA官方在2017GTC大会上首次提出：[PPT](https://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf)
+ NVIDIA官方INT8量化文档：[document](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#working-with-int8)
+ NVIDIA官方代码+示例samples：[code](https://github.com/NVIDIA/TensorRT/tree/master/samples)
+ NVIDIA官方在2022春季GTC大会演讲：[PPT](https://www.nvidia.com/en-us/on-demand/session/gtcspring22-s41440/)

```shell
# 方式一：在onnx转TensorRT时，使用trtexec工具量化
./trtexec --int8 --calib=<file>

# 方式2：使用python或C++的API量化
```

##### 2.2 QAT方式：TensoRT 8及之后版本
+ TensorRT-8可以显式地load包含有QAT量化信息的ONNX模型，实现一系列优化后，可以生成INT8的engine。


## 3. TensorRT提供的pytorch-quantization工具
+ 提供QAT和PTQ两种量化方式；
+ NVIDIA官方使用文档：[document](https://docs.nvidia.com/deeplearning/tensorrt/pytorch-quantization-toolkit/docs/userguide.html)
+ NVIDIA官方代码在TensorRT项目下的tool：[code](https://github.com/NVIDIA/TensorRT/tree/master/tools/pytorch-quantization)


### 4. Pytorch官方提供的量化工具
+ Pytorch官方量化工具文档：[document](https://pytorch.org/docs/stable/quantization.html)
+ torch.fx工具文档：[document](https://pytorch.org/docs/stable/fx.html)
+ Pytorch量化介绍博客：[blog](https://pytorch.org/blog/introduction-to-quantization-on-pytorch/)
+ Pytorch量化教程示例：[tutorial](https://pytorch.org/tutorials/recipes/quantization.html)


### 5. Torch-TensorRT工具
+ 2021年12月，由Nvidia和Meta共同提出；
+ Pytorch官方介绍文档：[document](https://pytorch.org/TensorRT/)
+ 项目github代码：[code](https://github.com/pytorch/TensorRT)


### 参考资料：
+ 知乎：TensorRT中的int8量化是怎样进行的？[链接](https://www.zhihu.com/question/421743958/answer/2428148997)
+ TensorRT的engine模型的结构图 [链接](https://mp.weixin.qq.com/s?__biz=Mzg3ODU2MzY5MA==&mid=2247488759&idx=1&sn=254c8c288bf3b87b80c47593d6e3b740&chksm=cf108cf2f86705e44616970ac9f4d2644c12f8063492906d9f0827913c6b1e71581f139f0322&token=1900338963&lang=zh_CN#rd)
