import numpy as np
import cv2
from matplotlib import pyplot as plt

#######################################################
# This is the task 6
#######################################################

def my_rotation(img, angle):
    row, col = img.shape
    newImg = np.zeros((row, col,3), np.uint8)
    for i in range(int(-row / 2), int(row / 2)):
        for j in range(int(-col / 2), int(col / 2)):
            xprime = np.cos(angle*np.pi/180) * i + np.sin(angle*np.pi/180) * j
            yprime = -np.sin(angle*np.pi/180) * i + np.cos(angle*np.pi/180) * j
            newx = int(xprime + row / 2)
            newy = int(yprime + col / 2)
            if(newx >= 0 and newx < row and newy >= 0 and newy < col):
                newImg[newx, newy] = img[int(i + row / 2), int(j + row / 2)]
    return newImg

# I am using forward mapping to rotate the image, which will leave a lot of holes in the image.
# In the lecture slide, it says splatting is needed after rotation to reduce holes.
# Here I am using a guassian filter to eliminate holes. (just like alleviate noises using Gaussian filter)

face3 = cv2.imread('face-03-U6656110.jpg')
face3 = cv2.cvtColor(face3, cv2.COLOR_BGR2GRAY)
face3 = cv2.resize(face3, dsize=(512, 512))
ang = [-90, -45, 15, 45, 90]
for i, a in enumerate(ang):
    rotated = my_rotation(face3, a)
    myrotated = cv2.GaussianBlur(rotated, (5, 5), 0)
    plt.imshow(myrotated, 'gray')
    plt.title("Rotate {} degree".format(a))
    plt.show()