# -*- coding: utf-8 -*-
# CLAB3 
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
#
# I = Image.open('stereo2012a.jpg')
#
# plt.imshow(I)
# uv = plt.ginput(6) # Graphical user interface to get 6 points

#####################################################################
def calibrate(im, XYZ, uv):
    points, _ = XYZ.shape
    # Initialize A and b
    A = np.zeros((2 * points, 11))
    b = np.zeros(2 * points)
    # Assign values to the corresponding matrices
    for i in range(points):
        A[2 * i] = [XYZ[i, 0], XYZ[i, 1], XYZ[i, 2], 1, 0, 0, 0, 0, -uv[i, 0] * XYZ[i, 0], -uv[i, 0] * XYZ[i, 1], -uv[i, 0] * XYZ[i, 2]]
        A[2 * i + 1] = [0, 0, 0, 0, XYZ[i, 0], XYZ[i, 1], XYZ[i, 2], 1, -uv[i, 1] * XYZ[i, 0], -uv[i, 1] * XYZ[i, 1], -uv[i, 1] * XYZ[i, 2]]
        b[2 * i] = uv[i, 0]
        b[2 * i + 1] = uv[i, 1]
    # Using np.linalg.lstsq to calculate the least square of Ax-b
    c, residual, rank, s = np.linalg.lstsq(A, b, rcond=None)
    # Add the last parameter to the matrix and reshape
    tmp = np.zeros(12)
    tmp[:11] = c
    tmp[11] = 1
    # Normalize the parameters
    normalized_c = tmp / np.sqrt(np.sum(tmp ** 2))
    C = np.zeros((3, 4))
    for j in range(3):
        C[j] = normalized_c[4 * j : 4 * j + 4]

    # Calculate the coordinates of the projected points
    homo = np.hstack((XYZ, np.ones((points, 1))))
    res = C @ homo.T
    px = res[0] / res[2]
    py = res[1] / res[2]
    # Calculate the mean square error of porjected points and uv points
    error = 0.0
    for k in range(points):
        temporary = np.array([px[k], py[k]]) - uv[k]
        error += np.linalg.norm(temporary)
    print('The mean squared error between uv and projected XYZ porints is: ' + str(error / points))
    print('The least squared error in satisfying the constraints is: ' + str(residual))
    np.save('Calibration', C)

    # Calculate the origin coordinates by project (0,0,0) to the image
    n = np.array([0, 0, 0, 1]).reshape((4, 1))
    origin = C @ n
    ox = origin[0] / origin[2]
    oy = origin[1] / origin[2]

    # Calculate the pixel coordinates of three axis
    x = np.array([100, 0, 0, 1]).reshape((4, 1))
    x_axis = C @ x
    xx = x_axis[0] / x_axis[2]
    xy = x_axis[1] / x_axis[2]

    y = np.array([0, 100, 0, 1]).reshape((4, 1))
    y_axis = C @ y
    yx = y_axis[0] / y_axis[2]
    yy = y_axis[1] / y_axis[2]

    z = np.array([0, 0, 100, 1]).reshape((4, 1))
    z_axis = C @ z
    zx = z_axis[0] / z_axis[2]
    zy = z_axis[1] / z_axis[2]

    # Draw the world coordinate system
    cv2.line(img, (ox, oy), (xx, xy), (255, 0, 0), 2)
    cv2.line(img, (ox, oy), (yx, yy), (255, 0, 0), 2)
    cv2.line(img, (ox, oy), (zx, zy), (255, 0, 0), 2)

    # Draw the original points and projected points
    #plt.plot(uv[:, 0], uv[:, 1], 'rx')
    #plt.plot(px, py, 'w.')
    #plt.imshow(im)
    #plt.show()
    return C


img = plt.imread('stereo2012a.jpg')
XYZ = np.load('XYZ2.npy')
uv = np.load('b2.npy')
calibrate(img, XYZ, uv)

'''
%% TASK 1: CALIBRATE
%
% Function to perform camera calibration
%
% Usage:   calibrate(image, XYZ, uv)
%          return C
%   Where:   image - is the image of the calibration target.
%            XYZ - is a N x 3 array of  XYZ coordinates
%                  of the calibration target points. 
%            uv  - is a N x 2 array of the image coordinates
%                  of the calibration target points.
%            K   - is the 3 x 4 camera calibration matrix.
%  The variable N should be an integer greater than or equal to 6.
%
%  This function plots the uv coordinates onto the image of the calibration
%  target. 
%
%  It also projects the XYZ coordinates back into image coordinates using
%  the calibration matrix and plots these points too as 
%  a visual check on the accuracy of the calibration process.
%
%  Lines from the origin to the vanishing points in the X, Y and Z
%  directions are overlaid on the image. 
%
%  The mean squared error between the positions of the uv coordinates 
%  and the projected XYZ coordinates is also reported.
%
%  The function should also report the error in satisfying the 
%  camera calibration matrix constraints.
% 
% your name, date 
'''

############################################################################
def homography(u2Trans, v2Trans, uBase, vBase):
    points = len(u2Trans)
    # Initialize matrices
    A = np.zeros((2 * points, 8))
    b = np.zeros((points * 2, 1))
    # Assign parameters to the matrix
    for i in range(points):
        A[2 * i] = np.array([u2Trans[i], v2Trans[i], 1, 0, 0, 0, -uBase[i] * u2Trans[i], -uBase[i] * v2Trans[i]])
        A[2 * i + 1] = np.array([0, 0, 0, u2Trans[i], v2Trans[i], 1, -vBase[i] * u2Trans[i], -vBase[i] * v2Trans[i]])
        b[2 * i] = uBase[i]
        b[2 * i + 1] = vBase[i]
    # Calculate the least square of AH-b
    c = np.linalg.lstsq(A, b, rcond=None)[0]
    # residual = np.linalg.lstsq(A, b, rcond=None)[1]
    tmp = np.zeros(9)
    tmp[:8] = c.flatten()
    tmp[8] = 1
    normalized_h = tmp / np.sqrt(np.sum(tmp ** 2))
    H = np.zeros((3, 3))
    # Convert the result to homography matrix
    for j in range(3):
        H[j] = normalized_h[3 * j: 3 * j + 3]
    np.save('H', H)
    return H


left = plt.imread('Left.jpg')
# plt.imshow(left)
# trans = plt.ginput(6) # Graphical user interface to get 6 points
# np.save('trans', trans)

# trans = np.load('trans.npy')
# print(trans)
# img = plt.imread('Left.jpg')
# plt.plot(trans[:, 0], trans[:, 1], 'rx')
# plt.imshow(img)
# plt.show()

right = plt.imread('Right.jpg')
# plt.imshow(right)
# plt.show()
# base = plt.ginput(6, timeout=0)
# np.save('base', base)
#
# base = np.load('base.npy')
# print(base)

trans = np.load('trans.npy')
base = np.load('base.npy')
u2Trans = trans[:, 0]
v2Trans = trans[:, 1]
uBase = base[:, 0]
vBase = base[:, 1]
# homography(u2Trans, v2Trans, uBase, vBase)

# H = np.load('H.npy')
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
H = np.load('H.npy')

warpped = np.zeros((left.shape), dtype=int)
row, col, chl = left.shape
# for i in range(row):
#     for j in range(col):
#         c = H @ np.array([[j], [i], [1]])
#         x = int(c[0] / c[2])
#         y = int(c[1] / c[2])
#         if y < row and x < col:
#             warpped[y, x] = left[i, j]

res = cv2.warpPerspective(left, H, (row, col))
plt.imshow(res)
plt.show()

'''
%% TASK 2: 
% Computes the homography H applying the Direct Linear Transformation 
% The transformation is such that 
% p = np.matmul(H, p.T), i.e.,
% (uBase, vBase, 1).T = np.matmul(H, (u2Trans , v2Trans, 1).T)
% Note: we assume (a, b, c) => np.concatenate((a, b, c), axis), be careful when 
% deal the value of axis 
%
% INPUTS: 
% u2Trans, v2Trans - vectors with coordinates u and v of the transformed image point (p') 
% uBase, vBase - vectors with coordinates u and v of the original base image point p  
% 
% OUTPUT 
% H - a 3x3 Homography matrix  
% 
% your name, date 
'''


############################################################################
def rq(A):
    # RQ factorisation

    [q,r] = np.linalg.qr(A.T)   # numpy has QR decomposition, here we can do it 
                                # with Q: orthonormal and R: upper triangle. Apply QR
                                # for the A-transpose, then A = (qr).T = r.T@q.T = RQ
    R = r.T
    Q = q.T
    return R,Q

