import numpy as np
import cv2
from matplotlib import pyplot as plt

#######################################################
# This is the task 4
#######################################################

# Crop the image with specified size
face2 = cv2.imread('face-02-U6656110.jpg')
f2 = cv2.cvtColor(face2, cv2.COLOR_BGR2RGB)
plt.subplot(1, 2, 1)
plt.imshow(f2)
plt.title('Original image')

crop = face2[100:450, 350:700]
crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
crop = cv2.resize(crop, dsize=(256, 256))
cv2.imwrite('Cropped.jpg', crop)
plt.subplot(1, 2, 2)
plt.imshow(crop, 'gray')
plt.title("Cropped image")
plt.tight_layout()
plt.show()

# Add gaussian noise to the image
gauss = 15 * np.random.randn(256, 256)
noisy = crop + gauss
plt.imshow(noisy, 'gray')

# Compute the histogram of the noisy image and the original image
plt.subplot(1, 2, 1)
plt.hist(crop.ravel(),10,[0,256]);
plt.title('Original histogram')

plt.subplot(1, 2, 2)
plt.hist(noisy.ravel(),10,[0,256]);
plt.title("After adding noise")
plt.tight_layout()
plt.show()

# My own implementation of gaussian filter
gauss = cv2.getGaussianKernel(5, 0)
gauss = gauss * gauss.transpose(1, 0)
def my_Gauss_filter(img, kernel):
    nImg = img.copy()
    row, col = img.shape
    for i in range(2, row - 2):
        for j in range(2, col - 2):
            t = img[i - 2: i + 3, j - 2: j + 3] * kernel
            nImg[i, j] = np.sum(t)
    return nImg

# Filter the noisy image using my own guassian filter
filtered = my_Gauss_filter(noisy, gauss)

plt.subplot(1, 2, 1)
plt.title("Noisy image")
plt.imshow(noisy, 'gray')

plt.subplot(1, 2, 2)
plt.title("Filtered image")
plt.imshow(filtered, 'gray')


plt.show()

# Compare my own gaussian filter with inbuilt gaussian filter

inbuiltFilter = cv2.GaussianBlur(noisy, (5, 5), 0)

plt.subplot(1, 2, 1)
plt.title('Inbuilt filter')
plt.imshow(inbuiltFilter, 'gray')


plt.subplot(1, 2, 2)
plt.title('My filter')
plt.imshow(filtered, 'gray')
plt.show()