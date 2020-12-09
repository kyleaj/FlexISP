import cv2
import numpy as np
import proximal
import typing

def get_bayer_decimators(x):
    r_channel = np.zeros(x.shape)
    r_channel[1::2, 1::2, 0] = 1
    g_channel = np.zeros(x.shape)
    g_channel[0::2, 1::2, 1] = 1
    g_channel[1::2, 0::2, 1] = 1
    b_channel = np.zeros(x.shape)
    b_channel[0::2, 0::2, 2] = 1

    return r_channel + g_channel + b_channel

def bayerify_proximal(x, mask):
    #return proximal.sum([proximal.mul_elemwise(r, x), proximal.mul_elemwise(g, x), proximal.mul_elemwise(b, x)])
    return proximal.mul_elemwise(mask,x) #+ proximal.mul_elemwise(g,x) + proximal.mul_elemwise(b,x)

def bayerify_numpy(x, mask):
    return x[:,:,:]*mask

def testDemosaic():
    test = np.arange(0, 12).reshape(2,2,3)

    decimators = get_bayer_decimators(test)
    bayer_np = bayerify_numpy(test, decimators)

    prox_fn = bayerify_proximal(proximal.Variable(test.shape), decimators)

    out = np.zeros_like(test).astype(float)

    prox_fn.forward([test], [out])

    print(bayer_np)
    print(out)