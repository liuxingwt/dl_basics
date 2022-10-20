## 量化工具介绍
#### 1.量化基础概念
（1）QAT和PTQ量化方式介绍：前者是Quantization Aware Training的简称，后者是Post Training Quantization的简称；
（2）


#### 1.TensorRT原生的INT8量化
+ 属于PTQ的方式；
+ NVIDIA官方，[PPT](https://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf)；
+ NVIDIA官方INT8量化文档，[document](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#working-with-int8);
+ NVIDIA官方代码+示例samples，[code]()https://github.com/NVIDIA/TensorRT/tree/master/samples

#### 1.2 TensorRT提供的pytorch-quantization工具
+ NVIDIA官方代码在TensorRT项目下的tool，[code](https://github.com/NVIDIA/TensorRT/tree/master/tools/pytorch-quantization);
+ 使用文档：[document](https://docs.nvidia.com/deeplearning/tensorrt/pytorch-quantization-toolkit/docs/userguide.html)


#### 2.1 Pytorch官方提供的量化工具
+ Pytorch官方量化工具文档：[document](https://pytorch.org/docs/stable/quantization.html)
+ torch.fx工具文档：[document](https://pytorch.org/docs/stable/fx.html)
+ Pytorch量化介绍博客：[blog](https://pytorch.org/blog/introduction-to-quantization-on-pytorch/)
+ Pytorch量化教程示例：[tutorial](https://pytorch.org/tutorials/recipes/quantization.html)

#### 2.2 Torch-TensorRT工具
+ 2021年12月，由Nvidia和Meta共同提出；
+ 项目github代码：[code](https://github.com/pytorch/TensorRT)
+ Pytorch官方文档：[document](https://pytorch.org/TensorRT/)
