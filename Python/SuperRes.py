import cv2
import numpy as np
import imutils
from Utils import get_super_res_shrink_matrix, flatten_by_channel, un_flatten_by_channel, get_identity_like, get_init_super_res
from scipy import sparse
from scipy.sparse import linalg
from scipy import signal
import pypher

iters = 10

im = imutils.resize(cv2.imread("Test Images/tampa.jpg"), height=300)
im = im[:300, :300, :]

super_shape = (600, 600, 3)

z = flatten_by_channel(im).astype(np.float)/255.0

up_im = get_init_super_res(im)
x_bar = x = flatten_by_channel(up_im) / 255.0
A = get_super_res_shrink_matrix(up_im)

phi0 = 0.1
phi1 = 1
phi2 = 0.25
n = 0.002

def K0(x): #gradient

    x = un_flatten_by_channel(x, super_shape)

    GradX = np.array([[-1, 1, 0]])
    GradY = GradX.T

    dX = cv2.filter2D(x, -1, GradX, borderType=cv2.BORDER_REPLICATE)
    dY = cv2.filter2D(x, -1, GradY, borderType=cv2.BORDER_REPLICATE)

    dX = flatten_by_channel(dX)[:, np.newaxis]
    dY = flatten_by_channel(dY)[:,np.newaxis]

    return np.concatenate((dX, dY), axis=1)

def K1(x):
    return x

def K0_transpose(x): # transpose gradient op
    x1 = un_flatten_by_channel(x[:, 0], super_shape)
    x2 = un_flatten_by_channel(x[:, 1], super_shape)

    GradX = np.array([[-1, 1, 0]]).T
    GradY = GradX.T

    dX = cv2.filter2D(x1, -1, GradX, borderType=cv2.BORDER_REPLICATE)
    dY = cv2.filter2D(x2, -1, GradY, borderType=cv2.BORDER_REPLICATE)

    return flatten_by_channel(dX+dY)

def K1_transpose(x):
    return x.flatten()

def L1NormProx(v, thresh):
    v_sign = np.sign(v)
    v = np.abs(v) - thresh
    v[v<0] = 0
    assert v.min() >= 0
    
    return (v*v_sign)+0

def NLMProx(v):
    # tonemap
    v_min = v.min()
    v_max = x.max()

    if ((v_min - v_max) == 0) and v_max == 0:
        return v * 0


    v = (v - v_min) / (v_max - v_min)
    v = v * 255
    v = v.astype(np.uint8)

    v = un_flatten_by_channel(v, super_shape)

    #cv2.imshow("Before NLM", v)

    v = cv2.fastNlMeansDenoisingColored(v)
    #cv2.imshow("After NLM", v)
    #cv2.waitKey(0)
    v = flatten_by_channel(v).astype(float) / 255.0

    v = v * (v_max - v_min)
    v = v + v_min

    return v

y = np.concatenate((K0(x), K1(x)[:, np.newaxis]), axis=1)

#mask z

gamma = 40
tau = 0.9/gamma
theta = 1

def penalty_op():
    tv_v = y[:, 0:2] + (gamma*K0(x_bar))
    nlm_v = y[:, 2] + (gamma*K1(x_bar))
    y[:, 0:2] = phi0*(tv_v - (gamma*L1NormProx(tv_v/gamma, 1/gamma)))
    y[:, 2] = phi1*(nlm_v - (gamma*NLMProx(nlm_v/gamma)))
    return y

def apply_K_transpose(x):
    return K0_transpose(x[:, 0:2]) + K1_transpose(x[:, 2])

def data_fidelity_op():
    left = A.T@A
    left = left*tau/n
    left = left + get_identity_like(left)

    v = x - (tau*apply_K_transpose(y))

    print(z.shape)
    print(A.T.shape)

    right = A.T@z
    right = right*tau/n
    right = right + v

    return linalg.cg(left, right)

def extrapolation(x_new, x_old):
    return x_new + (theta*(x_new - x_old))

initial = None
for _ in range(iters):
    y = penalty_op()
    x_old = x
    x, _ = data_fidelity_op()
    x_bar = extrapolation(x, x_old)

    print(x.min())
    print(x.max())
    print()

    reconstructed_im = un_flatten_by_channel(x, super_shape)
    reconstructed_im = (reconstructed_im-reconstructed_im.min())/(reconstructed_im.max()-reconstructed_im.min())
    reconstructed_im = reconstructed_im * 255
    reconstructed_im[reconstructed_im < 0] = 0
    reconstructed_im[reconstructed_im > 255] = 254
    reconstructed_im = reconstructed_im.astype(np.uint8)

    if (initial is None):
        initial = reconstructed_im.copy()

    cv2.imshow("x", reconstructed_im)
    cv2.waitKey(10)

print("Done")
cv2.imshow("x0", initial)
cv2.waitKey(0)