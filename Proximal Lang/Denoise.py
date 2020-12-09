from proximal import *
import numpy as np
import cv2
import imutils

import scipy.misc

input = cv2.imread("C:\\Users\\Kyle\\Pictures\\test.png")
input = scipy.misc.ascent()
input = input[:,:,np.newaxis].astype(np.uint8)
print(input.shape)
input = imutils.resize(input, width=300)
#input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)

b = input + 10*np.random.randn(*input.shape)

# Construct and solve problem.
x = Variable(input.shape)
prob = Problem(sum_squares(x - b/255) + .1 * norm1(grad(x)) + patch_NLM(x))
prob.solve(show_graph = True)

cv2.imwrite("Input.png", input)
cv2.imwrite("Noised.png", np.clip(b.astype(np.uint8), 0, 255))
print(np.max(x.value))
print(np.min(x.value))
cv2.imwrite("Output.png", np.clip(x.value*255, 0, 255).astype(np.uint8))