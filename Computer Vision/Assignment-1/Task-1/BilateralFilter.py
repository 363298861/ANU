import numpy as np
from matplotlib import pyplot as plt
import cv2

def my_bilateral_filter(img, w, sigma_s, sigma_r):
    w = w // 2
    y, x = np.ogrid[-w:w+1, -w:w+1]
    gaus = np.exp(-(x * x + y * y) / (2. * sigma_s * sigma_s))
    row, col = img.shape
    res = np.zeros((row, col))
    for i in range(0, row):
        for j in range(0, col):
            if i < w or j < w or i >= row - w or j >= col - w:
                res[i, j] = img[i, j]
                continue
            I = img[i - w : i + w + 1, j - w : j + w + 1]
            f = np.exp(-((I - img[i, j]) ** 2) / (2. * sigma_r * sigma_r))
            F = np.multiply(f, gaus)
            res[i, j] = np.sum(np.multiply(I, F)) / np.sum(F)
    return res

img = cv2.imread('opera.jpg', 0)
plt.imshow(img, 'gray')
#plt.show()

myimg = img.astype(int)
myimg = my_bilateral_filter(myimg, 9, 50, 50)
plt.imshow(myimg, 'gray')
plt.show()
#
# inbuilt = cv2.bilateralFilter(img, 9, 75, 75)
# plt.imshow(inbuilt, 'gray')
# plt.show()