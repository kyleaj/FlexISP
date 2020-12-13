import numpy as np
import scipy
from scipy import sparse
from scipy.sparse import linalg
import cv2
from Utils import flatten_by_channel, un_flatten_by_channel
import proximal

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

    def grad_transpose(self, v):
        #return v[:,0] + v[:,1]
        GradX = np.array([[-1, 1, 0]])
        GradY = GradX.T

        vx = un_flatten_by_channel(v[:,0],self.shape)
        vy = un_flatten_by_channel(v[:,1],self.shape)

        dX = cv2.filter2D(vx, -1, GradX.T, borderType=cv2.BORDER_REPLICATE)
        dY = cv2.filter2D(vy, -1, GradY.T, borderType=cv2.BORDER_REPLICATE)        

        return flatten_by_channel(dX) + flatten_by_channel(dY)

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

        D = np.concatenate((dX[:,np.newaxis], dY[:,np.newaxis]), axis=1)

        v = y + (self.gamma*D)
        v_orig = v

        v = v/self.gamma

        v = self.L1_norm_prox(v, 1/self.gamma)

        return v_orig - self.gamma*v

    def NLM(self, x, y):
        v = y + (self.gamma*x)
        v_orig = v.copy()
        v = v / self.gamma

        v = un_flatten_by_channel(v, self.shape)

        v_max = v.max()
        v_min = v.min()

        v = (v-v_min) * 255 / (v_max - v_min)

        prox = flatten_by_channel(cv2.fastNlMeansDenoisingColored(v.astype(np.uint8))).astype(float)/255
        prox = prox * (v_max - v_min)
        prox = prox + v_min

        return v_orig - (self.gamma*prox)

    def penalty(self, x, y):
        #return self.totalVariation(x, y)*0.1 + self.NLM(x,y) + self.crossChannel(x,y)
        return self.totalVariation(x,y)*0.1

    def applyKTranspose(self, y):
        return self.grad_transpose(y)
    

    def data_fidelity(self, x, y, z):
        v = x - self.t*self.applyKTranspose(y)

        left = self.A.T@self.A
        left = self.t/self.n*left
        rows = np.arange(left.shape[0], dtype=int)
        left = left + sparse.coo_matrix(([1]*left.shape[0], (rows, rows)), left.shape)

        right = self.A.T@z
        right = self.t/self.n * right
        right = right + v

        return linalg.cg(left, right)

    def extrapolation(self, x, x_new):
        return x + self.theta*(x_new - x)

    def forward(self, z, init_guess, iters):
        self.shape = z.shape
        x_bar = x = init_guess
        # For debugging
        answer = flatten_by_channel(z)
        z = self.A@flatten_by_channel(z)
        y = 0

        print("Starting iterations...")

        for _ in range(iters):
            y = self.penalty(x_bar, y)
            x_new, _ = self.data_fidelity(x, y, z)
            x_bar = self.extrapolation(x, x_new)
            x = x_new

            out = (x-x.min()) * 255 / (x.max() - x.min())
            cv2.imshow("intermediate", un_flatten_by_channel(out.astype(np.uint8), self.shape))
            cv2.waitKey(100)
            print((((out/255.0)**2 + (answer**2))**0.5).sum())
        return x