from Utils import *
from Base import FlexISP
import cv2
import numpy as np
from scipy import sparse
import scipy
import proximal

def test_bayerfication():
    dim = 512
    x = np.arange(0,dim*dim*3).reshape((dim,dim,3))

    decimator = get_bayer_decimator(x)
    bayer = flatten_by_channel(x*decimator)
    bayer = bayer[np.nonzero(bayer)]

    A = get_bayer_decimation_matrix(x)

    assert np.all(A@flatten_by_channel(x) == bayer)

def test_split_recombine():
    x = np.random.random((512,512,3))
    recombine = un_flatten_by_channel(flatten_by_channel(x), x.shape)
    assert np.all(x == recombine)

'''
def testGrad():
    f = FlexISP()

    test = np.array([[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]]).astype(float)
    test = cv2.imread("Test Images/GradTest.png").astype(float)/255
    #print(test)
    #print()

    f_out = f.grad(test)
    #print(f_out[0])
    #print(f_out[1])
    #print()

    proximal_fn = proximal.grad(proximal.Variable(test.shape), dims=2)
    prox_out = np.zeros((test.shape[0], test.shape[1], 3, 2))
    proximal_fn.forward([test], [prox_out])
    #print(prox_out[0].T)
    #print(prox_out[1].T)
    #print()

    print(prox_out[0].T.shape)
    print(f_out[0].shape)

    cv2.imshow("test", (np.abs(prox_out[:,:,:,1])*255).astype(np.uint8))
    cv2.waitKey(0)
'''

def testTotalVariation():
    im = scipy.misc.ascent()
    im = im[:,:,np.newaxis].astype(np.uint8)
    fn = proximal.norm1(proximal.grad(proximal.Variable(im.shape)))
    thresh=6

    out = fn.prox(thresh, im)
    

def testL1NormProx():
    f = FlexISP()

    v = np.arange(start=-12, stop=12, dtype=float)
    thresh = 6

    f_out = f.L1_norm_prox(v, thresh)

    prox_fn = proximal.norm1(proximal.Variable(v.shape))
    prox_out = prox_fn.prox(1/thresh, v)

    np.testing.assert_almost_equal(prox_out, f_out)

def main():
    #test_bayerfication()
    #test_split_recombine()
    #testGrad()
    #testL1NormProx()
    testTotalVariation()

if __name__=='__main__':
    main()