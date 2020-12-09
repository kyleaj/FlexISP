import proximal
from proximal import *
import cv2
from Utils import get_bayer_decimators, bayerify_numpy, bayerify_proximal
import imutils
import numpy as np
import time

input = cv2.imread("C:\\Users\\Kyle\\Pictures\\test.png")
input = imutils.resize(input, width=300)
input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)

mask = get_bayer_decimators(input)
bayer = bayerify_numpy(input, mask)

print("Compiling Problem...")
x = Variable(input.shape)
data_term = sum_squares(bayerify_proximal(x, mask) - (bayer / 255.0))

patch_similarity = patch_NLM(x)
grad_sparsity = norm1(grad(x))

objective = data_term + patch_similarity + grad_sparsity
#objective = data_term + grad_sparsity

p = Problem(objective)


print("Solving...")
start = time.time()
p.solve() #x0=input/255.0)
print("Done!")
print("Elapsed: ")
print(time.time() - start)
print()


cv2.imwrite("Input.png", input)
cv2.imwrite("Bayer.png", bayer)
print(np.max(x.value))
print(np.min(x.value))
print(np.average(x.value))
im = x.value.copy().astype(float)
im = (im - im.min())*255/(im.max()-im.min())
im = np.clip(im, 0, 255)
im = im.astype(np.uint8)
cv2.imwrite("Output.png", im)