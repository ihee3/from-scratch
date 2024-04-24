import numpy as np

class Dropout:
    def __init__(self, dropout_ratio = 0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flag = True):
        if train_flag:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (0.1 - self.dropout_ratio)
            
    def backward(self, dout):
         return dout * self.mask

def batchnorm_forward(x, gamma, beta, eps):
    N, D = x.shape

    mu = 1./N * np.sum(x, axis=0)
    xmu = x - mu

    sq = xmu ** 2
    var = 1./N * np.sum(sq, axis=0)

    sqrtvar = np.sqrt(var + eps)

    ivar = 1./ivar

    xhat = xmu * ivar

    gammax = gamma * xhat

    out = gammax + beta

    cache = (xhat, gamma, xmu, ivar, sqrtvar, var, eps)

    return out, cache

def batchnorm_backward(dout, cache):
    xhat, gamma, xmu, ivar, sqrtvar, var, eps = cache

    N, D = dout.shape

    dbeta = np.sum(dout, axis=0)
    dgammax = dout

    dgamma = np.sum(dgammax*xhat, axis =0)
    dxhat = dgamma * gamma

    divar = np.sum(dxhat * xmu, axis =0)
    dxmu_1 = dxhat * ivar

    dsqrtvar = -1. / (sqrtvar**2) * divar
    
    dvar = 1. / np.sqrt(var + eps) * dsqrtvar

    dsq = 1. /N *np.ones((N, D)) * dvar

    dxmu_2 = 2 * xmu * dsq

    dx1 = dxmu_1 + dxmu_2
    dmu = -1 * np.sum(dxmu_1+dxmu_2, axis = 0)

    dx2 = 1. / N * np.ones((N, D)) * dmu

    dx = dx1 + dx2

    return dx, dgamma, dbeta
