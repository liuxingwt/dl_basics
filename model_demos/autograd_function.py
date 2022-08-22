import torch


# 自定义的激活函数
class ReLU(torch.autograd.function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.clamp(min=0)
    
    @staticmethod
    def backward(ctx, grad_y):
        x, _ = ctx.saved_tensors
        grad_x = grad_y.clone()
        grad_x[x < 0] = 0
        return grad_x