## 解决方案：训练模型时出现loss=NaN
### 1. 初步筛查：某些算子的梯度可能会爆炸
+ log(x)：当x趋于0时，其导数趋于负无穷，通常需要用torch.clamp()操作确保输入不能太小。
+ exp(x)：易引起上溢出和下溢出。当函数值趋于0时，x会接近负无穷，此时可能会引起梯度爆炸。
+ actan(x1/x2)：括号内除法可能趋于无穷大，通常用atan(x1, x2)替代。

### 2. 检查梯度：使用Pytorch工具判断异常梯度
```python
import torch.autograd as autograd

# 反向传播时，当梯度异常时报错
# 如exp()算子求梯度异常，会报错"RuntimeError: Function ExpBackward"
with autograd.detect_anomaly():
    loss.backward()
```

### 3. 检查Tensor：前向传播时是否出现有Inf或NaN
```python
if torch.isnan(a).sum > 0:
    print("NaN exists in: ", a)
if torch.isinf(a).sum >0:
    print("Inf exists in: ", a)
```
