import numpy as np
import matplotlib.pyplot as plt
import cv2

left = plt.imread('Left.jpg')
# plt.imshow(left)
# trans = plt.ginput(6, timeout=0) # Graphical user interface to get 6 points
# np.save('trans', trans)

# trans = np.load('trans.npy')
# print(trans)
# img = plt.imread('Left.jpg')
# plt.plot(trans[:, 0], trans[:, 1], 'rx')
# plt.imshow(img)
# plt.show()
#
right = plt.imread('Right.jpg')
# plt.imshow(right)
# base = plt.ginput(4, timeout=0)
# np.save('base4', base)
# plt.show()
# base = np.load('base1.npy')
# print(base)


def homography(u2Trans, v2Trans, uBase, vBase):
    points = len(u2Trans)
    A = np.zeros((2 * points, 8))
    b = np.zeros((points * 2, 1))
    for i in range(points):
        A[2 * i] = np.array([u2Trans[i], v2Trans[i], 1, 0, 0, 0, -uBase[i] * u2Trans[i], -uBase[i] * v2Trans[i]])
        A[2 * i + 1] = np.array([0, 0, 0, u2Trans[i], v2Trans[i], 1, -vBase[i] * u2Trans[i], -vBase[i] * v2Trans[i]])
        b[2 * i] = uBase[i]
        b[2 * i + 1] = vBase[i]
    c = np.linalg.lstsq(A, b, rcond=None)[0]
    tmp = np.zeros(9)
    tmp[:8] = c.flatten()
    tmp[8] = 1
    H = np.zeros((3, 3))
    for j in range(3):
        H[j] = tmp[3 * j : 3 * j + 3]
    np.save('H', H)
    return H

trans = np.load('trans.npy')
base = np.load('base.npy')
u2Trans = trans[:, 0]
v2Trans = trans[:, 1]
uBase = base[:, 0]
vBase = base[:, 1]
homography(u2Trans, v2Trans, uBase, vBase)
H = np.load('H.npy')
# a = np.array([270, 117, 1]).reshape((3, 1))
# transa = H @ a
# print(transa)
# f, p = plt.subplots(1, 2)
# p[0].plot(u2Trans, v2Trans, 'r+')
# p[0].imshow(left)
# p[0].set_title('Points in the left image')
# p[1].plot(uBase, vBase, 'bx')
# p[1].imshow(right)
# p[1].set_title('Points in the right image')
# plt.show()
# H = np.load('H.npy')
# H = H / (np.sqrt(np.sum(H ** 2)))
# print(H)
# print(np.linalg.norm(H))
# # warpped = np.zeros((left.shape), dtype=int)

row, col, chl = left.shape
res = cv2.warpPerspective(left, H, (row, col))
plt.imshow(res)
plt.show()












