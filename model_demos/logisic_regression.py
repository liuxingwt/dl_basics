import numpy as np
import math

# 参数设置
N, D_in, D_out = 64, 100, 1
iteration = 100
lr = 1e-4

# 构造模型
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out) < 0.5
w1 = np.random.randn(D_in, D_out)
f = lambda x : 1.0 / (1 + math.exp(-x))

for i in range(iteration):
    # 前向传播
    logits = x.mm(w1)
    y_pred = f(logits)
    loss = (y_pred - y).pow(2).mean()

    # 反向传播计算梯度
    d_y_pred = 2 * (y_pred - y) / N
    d_logits = d_y_pred * y_pred * (1 - y_pred)
    d_w1 = x.Transpose().mm(d_logits)

    # 更新权重
    w1 -= lr * d_w1

