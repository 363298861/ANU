import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import cv2
# a = np.zeros((6, 3))
# a[0] = np.array([7, 7, 0])
# a[1] = np.array([14, 14, 0])
# a[2] = np.array([0, 7, 7])
# a[3] = np.array([0, 7, 14])
# a[4] = np.array([7, 0, 7])
# a[5] = np.array([14, 0, 14])
# np.save('XYZ', a)

# a = np.zeros((6, 3))
# a[0] = np.array([21, 7, 0])
# a[1] = np.array([7, 28, 0])
# a[2] = np.array([0, 7, 7])
# a[3] = np.array([0, 28, 21])
# a[4] = np.array([7, 0, 7])
# a[5] = np.array([21, 0, 21])
# np.save('XYZ2', a)

# b = np.load('XYZ.npy')
# print(b)

# I = Image.open('stereo2012a.jpg')
#
# plt.imshow(I)
# uv = plt.ginput(6) # Graphical user interface to get 6 points
#
# b = np.asarray(uv)
# print(b)
# np.save('b2', b)
# a = np.load('XYZ.npy')
# b = np.load('b.npy')
# print(b)
# x, residual, rank, s = np.linalg.lstsq(a, b, rcond=None)
# print(x)
# print(residual)


# print(b)


# c = np.load('Calibration.npy')

# point = np.array([7, 7, 0, 1])
# res = c @ point
# print(res)
#
# xyz = np.load('XYZ2.npy')
# uv = np.load('b2.npy')
# r = np.hstack((xyz, np.ones((6, 1))))
# C = np.load('Calibration.npy')
# res = C @ r.T
# px = res[0] / res[2]
# py = res[1] / res[2]
#
# n = np.array([0, 0, 0, 1]).reshape((4, 1))
# origin = C @ n
# ox = origin[0] / origin[2]
# oy = origin[1] / origin[2]
#
# x = np.array([100, 0, 0, 1]).reshape((4, 1))
# x_axis = C @ x
# xx = x_axis[0] /x_axis[2]
# xy = x_axis[1] /x_axis[2]
#
# y = np.array([0, 100, 0, 1]).reshape((4, 1))
# y_axis = C @ y
# yx = y_axis[0] /y_axis[2]
# yy = y_axis[1] /y_axis[2]
#
# z = np.array([0, 0, 100, 1]).reshape((4, 1))
# z_axis = C @ z
# zx = z_axis[0] / z_axis[2]
# zy = z_axis[1] / z_axis[2]
#
# img = plt.imread('stereo2012a.jpg')
# cv2.line(img, (ox, oy), (xx, xy), (255, 0, 0), 2)
# cv2.line(img, (ox, oy), (yx, yy), (255, 0, 0), 2)
# cv2.line(img, (ox, oy), (zx, zy), (255, 0, 0), 2)
# plt.plot(uv[:, 0], uv[:, 1], 'bx')
# plt.plot(px, py, 'r+')
# plt.imshow(img)
# plt.show()

# img = plt.imread('stereo2012a.jpg')
# plt.imshow(img)
# plt.show()

H = np.load('H.npy')
H = H / np.sqrt(np.sum(H ** 2))
print(H)