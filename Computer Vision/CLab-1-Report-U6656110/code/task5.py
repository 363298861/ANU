import numpy as np
import cv2
from matplotlib import pyplot as plt

#######################################################
# This is the task 5
#######################################################

# This is my own Sobel filter function
def my_Sobel_filter(img):
    gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    nImg = img.copy()
    row, col = img.shape
    for i in range(1, row - 1):
        for j in range(1, col - 1):
            x = img[i - 1 : i + 2 , j - 1: j + 2] * gx
            y = img[i - 1 : i + 2 , j - 1: j + 2] * gy
            px = np.sum(x)
            py = np.sum(y)
            nImg[i, j] = np.sqrt(px * px + py * py)
    return nImg

# Plot the Sobel filtered image with inbuilt function
crop = cv2.imread('Cropped.jpg', 0)
grad_x = cv2.Sobel(crop, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(crop, cv2.CV_64F, 0, 1, ksize=3)

abs_grad_x = cv2.convertScaleAbs(grad_x)
abs_grad_y = cv2.convertScaleAbs(grad_y)

grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

# Calculate my own Sobel filter image
crop1 = cv2.GaussianBlur(crop, (5, 5), 0)
my_sobel = my_Sobel_filter(crop1)

plt.subplot(1, 2, 1)
plt.imshow(np.uint8(my_sobel), 'gray')
plt.title('My Sobel filter')

plt.subplot(1, 2, 2)
plt.imshow(grad, 'gray')
plt.title('Inbuilt Sobel filter')

plt.show()