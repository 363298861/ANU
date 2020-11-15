import numpy as np
import cv2
from matplotlib import pyplot as plt

#######################################################
# This is the task 2
#######################################################

# Find the negative image
img = cv2.imread('Lenna.png', 0)
plt.subplot(1, 2, 1)
plt.imshow(np.uint8(img), 'gray')
plt.title('Grayscale image')

img2 = cv2.bitwise_not(img)
plt.subplot(1, 2, 2)
plt.imshow(np.uint8(img2), 'gray')
plt.title('Negative image')

plt.show()

# Flip the image vertically
img = cv2.imread('Lenna.png')
img_flipped = cv2.flip(img, 0)
img_flipped = cv2.cvtColor(img_flipped, cv2.COLOR_BGR2RGB)
plt.imshow(img_flipped)
plt.title('Vertically flipped image')
plt.show()

# Swap red and blue channel
img = cv2.imread('Lenna.png')
plt.imshow(img)
plt.title('Red and blue channel swapped')
plt.show()

# Average the two images above
img = cv2.imread('Lenna.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = (img + img_flipped) / 2
plt.imshow(np.uint8(img))
plt.title('Average two images')
plt.show()

# Add random value
rand = np.random.randint(0, 256)
img = cv2.imread('Lenna.png', 0)
img[img > 255 - rand] = 255

plt.imshow(img, cmap='gray')
plt.title('Add random value')
plt.show()




