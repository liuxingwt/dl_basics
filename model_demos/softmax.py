import numpy as np
import math

N, D_in, D_out = 64, 1000, 10

x = np.random.randn(N, D_in)
y = np.random.randint(0, 10, N)

w = np.random.randn(D_in, D_out)
f_softmax = lambda x : 1.0 / (1 + math.exp(-x))


iteration = 100
lr = 1e-5

for i in iteration:
