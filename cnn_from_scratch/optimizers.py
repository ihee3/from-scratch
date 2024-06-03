import numpy as np

class SGD:
    def __init__(self, lr = 0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr_grad[key]

class Momentum:
    def __init__(self, lr=0.01, momentum = 0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum*self.v[key] - self.lr*grads[key]
            params[key] += self.v[key]

class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)

class Adam:
    def __init__(self, lr, momentum_1=0.9, momentum_2=0.999):
        self.lr = lr
        self.momentum_1 = momentum_1
        self.momentum_2 = momentum_2
        self.v = None
        self.s = None

    def update(self, params, grads):
        if self.v == None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = (self.momentum_1*self.v[key] - self.lr*grads[key]) / (1 - self.momentum_1)
        '''
        v[key] = Momentum.update(params, grad)
        v[key] = v[key] / (1-self.momentum_1)
        '''

        if self.s == None:
            for key, val in params.items():
                self.s[key] = np.zeros_like(val)

        for key, val in params.items():
            self.s[key] = (self.momentum_2 * self.s[key] - self.lr * (grads[key]**2) ) / (1 - self.momentum_2)
            params[key] -= self.lr * self.v[key] / (np.sqrt(self.s[key]) + 1e-8)

