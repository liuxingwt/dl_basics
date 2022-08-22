import numpy as np


class BatchNormalization():
    def __init__():
        self.epsilon = 1e-5
        self.beta = 0.1
        self.gamma = 0.1
        pass
    
    def forward(x):
        """
        Args:
        - x.shape: N X C X H X W
        Return:
        """
        x_mean = x.mean(axis=(0,2,3))   # 1 X C X 1 X 1
        x_var = x.var(axis=(0,2,3))    # 1 X C X 1 X 1

        x_new = (x - x_mean) / (x_var + self.epsilon).pow(0.5)

        y = self.beta * x_new + self.gamma
        return y

    def backward(d_y, y):

        d_beta = d_y * 
        d_gamma = 
        d_x = sef.beta * y / (x_var + self.epsilon).pow(0.5)
        return 

