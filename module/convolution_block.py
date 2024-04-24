import sys, os
sys.path.append(os.pardir)
from utility import *
import numpy as np


class Pooling:
    def __init__(self, pool_h, pool_w, stride = 1, pad =0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride 
        self.pad = pad

    def forward(self, x):
        N, C, H, W = x.shape
        out_w = int(1+ ( H-self.pool_h)/self.stride)
        out_h = int(1 + (W - self.pool_w)/self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        out = np.max(col, axis=1)

        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        return out

  class Convolution:
    def __init__(self, W, b, stride = 1, pad=1):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

        self.x = None
        self.col = None
        self.col_w = None

        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = int(1 +(H +2*self.pad - FH)/self.stride)
        out_w = int(1+ (W + 2* self.pad - FW)/self.stride)

        col = im2col(x, FH, self.stride, self.pad)
        col_w = self.W.reshape(FN,  -1).T
        out = np.dot(col, col_w) + self.b

        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_w = col_w

        return out
    
    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        self.db = np.sum(dout, axis = 0)
        self.dxW = dout
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)
        dcol = np.dot(dout, self.col_w.T)
        dx = None

        return dx 
