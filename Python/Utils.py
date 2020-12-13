import cv2
import numpy as np
from scipy import sparse

def flatten_by_channel(x):
    r = x[:,:,0].flatten()
    g = x[:,:,1].flatten()
    b = x[:,:,2].flatten()

    return np.concatenate((r, g, b))

def un_flatten_by_channel(x, shape):
    c_len = shape[0]*shape[1]
    r = x[0:c_len].reshape(shape[0:2])
    g = x[c_len:2*c_len].reshape(shape[0:2])
    b = x[2*c_len:].reshape(shape[0:2])

    return cv2.merge((r, g, b))

def un_flatten_by_channel_super_res(x, shape):
    c_len = shape[0]*shape[1]
    print(c_len)
    r = x[0:c_len].reshape(shape[0:2])
    g = x[c_len:2*c_len].reshape(shape[0:2])
    b = x[2*c_len:].reshape(shape[0:2])

    return cv2.merge((r, g, b))

def get_bayer_decimator(x):
    r_channel = np.zeros(x.shape)
    r_channel[1::2, 1::2, 0] = 1
    g_channel = np.zeros(x.shape)
    g_channel[0::2, 1::2, 1] = 1
    g_channel[1::2, 0::2, 1] = 1
    b_channel = np.zeros(x.shape)
    b_channel[0::2, 0::2, 2] = 1

    return r_channel + g_channel + b_channel

def get_bayer_decimation_matrix(x):
    # Construct A
    data = []
    rows = []
    cols = []

    curr_row = 0

    h, w, c = x.shape
    # First the reds
    i = 0
    while i < h:
        i = i + 1 # Odd rows
        j = 0
        while j < w:
            j = j + 1 # odd columns
            data.append(1)
            rows.append(curr_row)
            cols.append((w*i)+j)
            curr_row = curr_row + 1
            j = j + 1
        i = i + 1

    # Greens
    i = 0
    while i < h:
        j = 0
        if i % 2 == 0:
            while j < w:
                j = j + 1 # odd columns
                data.append(1)
                rows.append(curr_row)
                cols.append((w*i)+j+h*w)
                curr_row = curr_row + 1
                j = j + 1
        else:
            while j < w:
                data.append(1)
                rows.append(curr_row)
                cols.append((w*i)+j+h*w)
                curr_row = curr_row + 1
                j = j + 2 # even columns
        i = i + 1

    # Blues
    i = 0
    while i < h:
        j = 0
        while j < w:
            data.append(1)
            rows.append(curr_row)
            cols.append((w*i)+j+h*w*2)
            curr_row = curr_row + 1
            j = j + 2 # Even columns
        i = i + 2 # Even rows 
    
    return sparse.coo_matrix((data, (rows, cols)), (h*w, h*w*c))

def get_identity_like(A):
    data = [1] * A.shape[0]
    rows = cols = np.arange(start=0, stop=A.shape[0])

    return sparse.coo_matrix((data, (rows, cols)), A.shape)

def get_init_demosaic(A):
    kernel = np.ones((3,3))

    decim = get_bayer_decimator(A)

    A = A*decim

    a = cv2.filter2D(A, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    b = cv2.filter2D(decim, -1, kernel, borderType=cv2.BORDER_CONSTANT)

    a = (a.astype(float) / b.astype(float)).astype(np.uint8)

    cv2.imshow("Init guess", a)
    cv2.waitKey(0)

    return a

def get_super_res_shrink_matrix(x):
    # Construct A, which selects bottom right of each quad of pixels
    data = []
    rows = []
    cols = []

    curr_row = 0

    h, w, c = x.shape
    # First the reds
    i = 0
    while i < h:
        i = i + 1 # Odd rows
        j = 0
        while j < w:
            j = j + 1 # odd columns
            data.append(1)
            rows.append(curr_row)
            cols.append((w*i)+j)
            curr_row = curr_row + 1
            j = j + 1
        i = i + 1

    # Greens
    i = 0
    while i < h:
        i = i + 1
        j = 0
        while j < w:
            j = j + 1 # odd columns
            data.append(1)
            rows.append(curr_row)
            cols.append((w*i)+j+h*w)
            curr_row = curr_row + 1
            j = j + 1
        i = i + 1

    # Blues
    i = 0
    while i < h:
        i = i + 1
        j = 0
        while j < w:
            j = j + 1 # odd columns
            data.append(1)
            rows.append(curr_row)
            cols.append((w*i)+j+h*w*2)
            curr_row = curr_row + 1
            j = j + 1
        i = i + 1
    
    return sparse.coo_matrix((data, (rows, cols)), (h*w*c//4, h*w*c))

def get_init_super_res(x):
    a = cv2.resize(x, (x.shape[0]*2, x.shape[1]*2))
    cv2.imshow("Init guess", a)
    cv2.waitKey(0)
    return a

'''
a = np.arange(12*2*2).reshape((4,4,3))
shrink = get_super_res_shrink_matrix(a)

print(a)
#print(flatten_by_channel(a))
print(shrink.todense()@flatten_by_channel(a))
res = shrink.todense()@flatten_by_channel(a)
res = res.flatten()[0,:].T
print(res.shape)
print(un_flatten_by_channel(res, (2,2)))
'''
