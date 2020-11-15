import numpy as np
import cv2
from matplotlib import pyplot as plt

#######################################################
# This is the task 3
#######################################################

# Resize my image
face1 = cv2.imread('face-01-U6656110.jpg')
face1 = cv2.cvtColor(face1, cv2.COLOR_BGR2RGB)
face1 = cv2.resize(face1, dsize=(720, 480))
plt.imshow(np.uint8(face1))
plt.title('Resized image')
plt.show()

# Find the grayscale image from R, G, B channel, respectively.
red, green, blue = cv2.split(face1)
plt.subplot(1, 3, 1)
plt.imshow(red, 'gray')
plt.title('Red channel')

plt.subplot(1, 3, 2)
plt.imshow(green, 'gray')
plt.title('Green channel')

plt.subplot(1, 3, 3)
plt.imshow(blue, 'gray')
plt.title('Blue channel')

plt.tight_layout()
plt.show()

# Compute their histogram
plt.hist(red.ravel(),10,[0,256]);
plt.title('Red channel histogram')
plt.show()

plt.hist(green.ravel(),10,[0,256]);
plt.title('Green channel histogram')
plt.show()

plt.hist(blue.ravel(),10,[0,256]);
plt.title('Blue channel histogram')

plt.show()

# Histogram equalization of 3 channels and the original image

equRed = cv2.equalizeHist(red)
equGreen = cv2.equalizeHist(green)
equBlue = cv2.equalizeHist(blue)

plt.subplot(2, 2, 1)
plt.title('Red channel equalized image')
plt.imshow(equRed, 'gray')

plt.subplot(2, 2, 2)
plt.title('Green channel equalized image')
plt.imshow(equGreen, 'gray')

plt.subplot(2, 2, 3)
plt.title('Blue channel equalized image')
plt.imshow(equBlue, 'gray')

res = cv2.merge([equRed, equGreen, equBlue])
plt.subplot(2, 2, 4)
plt.title('Histogram equalization image')
plt.imshow(res)

plt.tight_layout()
plt.show()