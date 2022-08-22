############ 使用 numpy 实现一个两层神经网络：前向+反向+计算梯度 ############
from pickletools import optimize
import numpy as np

# activate = lambda x : 1.0 / (1 + exp(-x))



############## 使用pytorch实现一个2层神经网络：前向+反向计算梯度+更新参数 ############
############ CS231 Lecture 6 P59 ##########################
import torch.nn as nn

class Neural_Network():
    def __init__(self, ):
        # Hyper parameters
        N, D_in, D_h, D_out = 64, 1000, 100, 10
        # input 
        x = torch.randn(N, D_in)
        y = torch.randn(N, D_out)
        # layer setting
        w1 = torch.randn(D_in, D_h, requires_grad=True)
        w2 = torch.randn(D_h, D_out, requires_grad=True)

    def forward(self, iteration = 100):
        for i in range(iteration):
            ## 自动求导，mm是矩阵乘法matrix muitiplication，
            ## 与此相对的mul是按元素相乘，element_wise 或者Hadamard乘积
            y_pred = x.mm(w1).clamp(min=0).mm(w2)
            self.loss = (y_pred - y).square().sum()
            ## 手动前向传播
            h = x.mm(w1)
            x1 = h.clamp(min=0)
            y_pred = x1.mm(w2)
            self.loss = (y_pred - y).pow(2).sum()
            # 手动求导
            grad_y_pred = 2 * (y_pred - y)
            grad_w2 = x1.T().mm(grad_y_pred)
            grad_x1 = grad_y_pred.mm(w2.T())
            grad_h = grad_x1.clone()
            grad_h[h < 0] = 0
            grad_w1 = x.T().mm(grad_h)

    def backward(self, lr=0.01):
        self.loss.backward()
        with torch.no_grad():
            w1 -= lr * w1.grad
            w2 -= lr * w2.grad
            w1.grad.zero_()          # 清零梯度，不然会inplace累加。累加可以加大batch size
            w2.grad.zero_()
 #########################################################################################       



############ 完全用Pytorch实现一个2层神经网络 ###############
############ CS231 Lecture 6 P71 ##########################
N, D_in, D_h, D_out = 64, 1000, 100, 10
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)


class TwoLayerNet():
    def __init__(self, D_in, D_h, D_out):
        self.linear1 = nn.Linear(D_in, D_h)
        self.act1 = nn.ReLu()
        self.linear2 = nn.Linear(D_h, D_out)
    
    def forward(self, x):
        out = self.linear2(self.act1(self.linear1(x)))
        return out

model = TwoLayerNet(D_in, D_h, D_out)

iteration = 1000
epochs = 100
lr = 0.001
# optimizer = torch.optim.Adam(params=model.parameters, lr=0.01)

train_dataset = 
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64)

for epoch in range(epochs):
    for x,y in train_loader:
        y_pred = model(x)
        loss = nn.functional.mse_loss(y, y_pred)

        loss.backward()
        # optimizer.step()
        # optimizer.zero_grad()

        with torch.no_grad():
            for param in model.parameters():
                param -= lr * param.grad
            model.zero_grad()
############################################################
