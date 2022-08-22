import numpy as np


class kMeans():
    def __init__():
        pass

    def train(x, y):
        self.x = x        # n_train x D_in
        self.y = y        # n_train 

    def infer(batch, k=1, p=2):
        res = []    
        for i in batch:    # i.shape: 1 x D_in
            dists = (i - self.x).norm(axis=1, p)   # n_train x 1
            idx = np.argmin(dists)
            out = self.y[idx]
            res.append(out)
        return res
