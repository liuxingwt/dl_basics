
## Pytorch转onnx模型中的问题

#### 1. torch.onnx.export(opset_version)与onnxruntime的版本对应关系

+ 参考1：https://onnxruntime.ai/docs/reference/compatibility.html#onnx-opset-support
+ 参考2：https://github.com/microsoft/onnxruntime/blob/main/docs/Versioning.md
+ 参考3：https://github.com/onnx/onnx/blob/main/docs/Versioning.md

#### 2.使用onnxruntime的python代码示例
+ 参考1：https://github.com/microsoft/onnxruntime-inference-examples/blob/main/python/api/onnxruntime-python-api.py

#### 3. pytorch模型转为onnx
```shell
# 首先加载pytorch模型和参数文件到cpu中
# 使用torch.onnx.export()函数将pytorch模型转换onnx格式
# 本例将这个函数封装到了pytorch2onnx.py文件中，直接执行下面命令可得到onnx格式的pfld模型
python3 pytorch2onnx.py

# 使用onnx.checker.check_model()函数检查转换结果是否正确
# 在check onnx model时报错：Segmentation fault (core dumped)； 
# 解决方法：先import onnx，再import torch

# 可以考虑使用onnx-simplifier压缩模型
pip3 install onnx
pip3 install onnx-simplifier

# 安装onnxruntime框架, 使用onnxruntime框架测试onnx格式模型
pip3 install onnxruntime

# 给定一个图片输入，检查onnx模型是否转换成功
import onnxruntime
```
