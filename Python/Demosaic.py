from Base import FlexISP
import cv2
from Utils import *
import scipy
import imutils

#im = cv2.imread("Test Images/GradTest.png")
#im = cv2.imread("Test Images/squares.jpg")
im = cv2.imread("Test Images/tampa.jpg")
im = imutils.resize(im, width=310, height=310)
im = im[:300, :300, :]
cv2.imshow("in", im.astype(np.uint8).reshape(im.shape))
cv2.waitKey(10)

f = FlexISP()
f.A = get_bayer_decimation_matrix(im)

init_guess = flatten_by_channel(im)*0
cv2.imshow("init_guess", un_flatten_by_channel(init_guess, im.shape).astype(np.uint8))
cv2.waitKey(10)
cv2.imshow("mosaic", (get_bayer_decimator(im)*im).astype(np.uint8))
cv2.waitKey(10)
z = im

out = f.forward(z/255.0, init_guess/255.0, 500)

out = (out-out.min()) * 255 / (out.max() - out.min())
cv2.imshow("out", un_flatten_by_channel(out.astype(np.uint8), im.shape))
cv2.waitKey(0)

cv2.imwrite("mosaic.png", (get_bayer_decimator(im)*im).astype(np.uint8))
cv2.imwrite("out.png", un_flatten_by_channel(out.astype(np.uint8), im.shape).astype(np.uint8))