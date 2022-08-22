## 解决方案：训练模型时出现loss=NaN
### 1. 初步筛查：常见的会引起溢出的算子
+ log(x)：当x趋于0时，其导数趋于负无穷，通常需要用torch.clamp(x, min=1e-6)操作确保输入不能太小。
+ exp(x)：易引起上溢出和下溢出。有可能是脏数据引起的exp(x)=0，此时x会接近负无穷，引起梯度爆炸。
+ actan(x1/x2)：括号内除法可能趋于无穷大，通常用atan(x1, x2)替代。

### 2. 检查梯度：使用Pytorch工具判断异常梯度
##### 2.1 查找原因
```python
import torch.autograd as autograd

# 反向传播时，当梯度异常时报错
# 如exp()算子求梯度异常，会报错"RuntimeError: Function ExpBackward"
with autograd.detect_anomaly():
    loss.backward()
```
##### 2.2 其他原因
+ 学习率过大？？？

##### 2.3 解决方法
+ 使用gradient clip
```python
from torch.nn.utils import clip_grad

# 反向传播求梯度
loss.backward()                        
params = model.parameters()
# 裁剪参数的梯度
clip_grad.clip_grad_norm_(filter(lambda p: p.requires_grad, params), **self.grad_clip)
# 使用优化算法，更新参数
optimizer.step()                        
```


### 3. 检查Tensor：前向传播时是否出现有Inf或NaN
```python
if torch.isnan(a).sum > 0:
    print("NaN exists in: ", a)
if torch.isinf(a).sum >0:
    print("Inf exists in: ", a)
```
