"""
CLAB Task-1: Harris Corner Detector
Your name (Your uniID): u6656110
"""

import numpy as np
import cv2


def conv2(img, conv_filter):
    # flip the filter
    f_siz_1, f_size_2 = conv_filter.shape
    conv_filter = conv_filter[range(f_siz_1 - 1, -1, -1), :][:, range(f_siz_1 - 1, -1, -1)]
    pad = (conv_filter.shape[0] - 1) // 2
    result = np.zeros((img.shape))
    img = np.pad(img, ((pad, pad), (pad, pad)), 'constant', constant_values=(0, 0))
    filter_size = conv_filter.shape[0]
    for r in np.arange(img.shape[0] - filter_size + 1):
        for c in np.arange(img.shape[1] - filter_size + 1):
            curr_region = img[r:r + filter_size, c:c + filter_size]
            curr_result = curr_region * conv_filter
            conv_sum = np.sum(curr_result)  # Summing the result of multiplication.
            result[r, c] = conv_sum  # Saving the summation in the convolution layer feature map.

    return result


def fspecial(shape=(3, 3), sigma=0.5):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


# Parameters, add more if needed
sigma = 2
thresh = 0.01
k = 0.04
# Derivative masks
dx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
dy = dx.transpose()
import matplotlib.pyplot as plt

bw = plt.imread('Harris_2.pgm')
#bw = cv2.cvtColor(bw, cv2.COLOR_RGB2GRAY)
bw = np.array(bw * 255, dtype=int)
#computer x and y derivatives of image
Ix = conv2(bw, dx)
Iy = conv2(bw, dy)

g = fspecial((max(1, np.floor(3 * sigma) * 2 + 1), max(1, np.floor(3 * sigma) * 2 + 1)), sigma)
Iy2 = conv2(np.power(Iy, 2), g)
Ix2 = conv2(np.power(Ix, 2), g)
Ixy = conv2(Ix * Iy, g)

######################################################################
# Task: Compute the Harris Cornerness
######################################################################
row, col = bw.shape
res = np.zeros((row, col))
R = np.zeros((row, col))
rmax = 0
for i in range(row):
    for j in range(col):
        M = np.array([[Ix2[i, j], Ixy[i, j]], [Ixy[i, j], Iy2[i, j]]], dtype=np.float64)
        R[i, j] = np.linalg.det(M) - k * np.power(np.trace(M), 2)
        if R[i, j] > rmax:
            rmax = R[i, j]

######################################################################
# Task: Perform non-maximum suppression and
#       thresholding, return the N corner points
#       as an Nx2 matrix of x and y coordinates
######################################################################
for i in range(1, row - 1):
    for j in range(1, col - 1):
        if R[i, j] > thresh * rmax and R[i, j] == np.amax(R[i-1 : i+2, j-1 : j+2]):
            res[i, j] = 1

cx, cy = np.where(res == 1)
plt.plot(cy, cx, 'r+')
bw = plt.imread('Harris_2.pgm')
plt.imshow(bw)
plt.show()


# Detect corners using built-in function cv2.cornerHarris()
# bw = plt.imread('Harris_1.jpg')
# img = cv2.imread('Harris_1.jpg')
# bw = cv2.cvtColor(bw, cv2.COLOR_RGB2GRAY)
# bw = np.float32(bw)
# dst = cv2.cornerHarris(bw, 2, 3, 0.04)
# #result is dilated for marking the corners, not important
# dst = cv2.dilate(dst,None)
# # Threshold for an optimal value, it may vary depending on the image.
# img[dst>0.01*dst.max()]=[255,0,0]
# plt.imshow(img, 'gray')
# plt.show()