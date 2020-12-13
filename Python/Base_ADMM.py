import numpy as np
import scipy
from scipy import sparse
from scipy.sparse import linalg
import cv2
from Utils import flatten_by_channel, un_flatten_by_channel

class FlexISP:
    def __init__(self, gamma=40, theta=1, n=0.002):
        self.gamma = gamma
        self.t = 1 # --> calculate
        self.theta = theta
        self.A = None
        self.M = None
        self.n = n

    def crossChannel(self, x, y):
        return 0

    def grad(self, v):
        GradX = np.array([[-1, 1, 0]])
        GradY = GradX.T

        dX = cv2.filter2D(v, -1, GradX, borderType=cv2.BORDER_REPLICATE)
        dY = cv2.filter2D(v, -1, GradY, borderType=cv2.BORDER_REPLICATE)

        return (dX, dY)

    def L1_norm_prox(self, v, thresh):
        v_sign = np.sign(v)
        v = np.abs(v) - thresh
        v[v<0] = 0
        return (v*v_sign)+0


    def totalVariation(self, x, y):
        x = un_flatten_by_channel(x, self.shape)

        dX, dY = self.grad(x)

        dX = flatten_by_channel(dX)
        dY = flatten_by_channel(dY)

        D = np.concatenate((dX, dY))

        v = y + (self.gamma*flatten_by_channel(D))
        v_orig = v

        v = v/self.gamma

        v = self.L1_norm_prox(v, 1/self.gamma)

        return v_orig - self.gamma*v

    

    def x_update(self, x, b, C, v):
        

    def forward(self, orig_im, A, iters):
        x = np.zeros_like(orig_im)
        z = np.concatenate((x.copy, x.copy), dim=-1)
        u = z.copy()

        for _ in range(iters):
            x = x_update(x, orig_im, A, z-u)
            z = z_update(x, u)
            u = u + grad(X) - z