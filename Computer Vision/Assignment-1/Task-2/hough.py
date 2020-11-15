# Partial code from: Alyssa Quek
#
# Assignment-1: Hough Transform
# Your name (Your uniID)
# Zhiyuan Huang U6656110
import numpy as np
import cv2
import math
from matplotlib import pyplot as plt

def hough_line(img, theta_step=1):
    """
    Hough transform for lines
    Input:
    img - 2D binary image with nonzeros representing edges
    angle_step - Spacing between angles to use every n-th angle
                 between -90 and 90 degrees. Default step is 1.
    Returns:
    accumulator - 2D array of the hough transform accumulator
    rhos - array of rho values. Max size is 2 times the diagonal
           distance of the input image. [-diag_len, diag_len]
    theta - array of angles used in computation, in radians. [-pi/2, pi/2]
    """
    # Rho and Theta ranges
    thetas = np.deg2rad(np.arange(-90.0, 90.0, theta_step))
    width, height = img.shape
    diag_len = int(round(math.sqrt(width * width + height * height)))   
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2)

    # Cache some resuable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Hough accumulator array of theta vs rho
    accumulator = np.zeros((2 * diag_len, num_thetas), dtype=np.uint8)
    # (row, col) indexes to edges
    y_idxs, x_idxs = np.nonzero(img)

    # Vote in the hough accumulator
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]

        for t_idx in range(num_thetas):
            # Calculate rho and add diag_len to it to make it a positive index
            r = int(round(x * cos_t[t_idx] + y * sin_t[t_idx]))
            ir = r + diag_len
            accumulator[ir, t_idx] += 1

    return accumulator, rhos, thetas

def peak_votes(accumulator, rhos, thetas, threshold=200):
    """ Finds the index with max number of votes in the hough accumulator """
    #idx = np.argmax(accumulator)
    # Here I modified the code from finding the maximum to
    # find all elements that are greater than the threshold and
    # stored them in a array
    indices = np.argwhere(accumulator > threshold)
    lines = indices.shape[0]
    rho = np.zeros(lines)
    theta = np.zeros(lines)
    for l in range(lines):
        rho[l] = rhos[indices[l, 0]]
        theta[l] = thetas[indices[l, 1]]
    #rho = rhos[int(idx / accumulator.shape[1])]
    #theta = thetas[idx % accumulator.shape[1]]
    return rho, theta

def draw(img, rho, theta):
    """ Draws the line in image """
    for i in range(rho.shape[0]):
        a = np.cos(theta[i])
        b = np.sin(theta[i])
        x0 = a * rho[i]
        y0 = b * rho[i]
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv2.line(img, pt1, pt2, (0,0,255), 2)

if __name__ == '__main__':
    filename = 'source.jpg'
    # Loads an image
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print('{0} not found!'.format(filename))
        exit()
    #cv2.imshow('Source', img)
    
    # Copy edges to the images that will display the results in BGR
    line_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Edge detection
    edge_img = cv2.Canny(img, 50, 150, None, 3)
    cv2.imwrite('edges-detected-{0}'.format(filename), edge_img)
    #cv2.imshow('Edges', edge_img)

    # Detect lines
    accumulator, rhos, thetas = hough_line(edge_img, 2)
    
    # Get the parameters of the line that got maximum votes (longest line)
    rho, theta = peak_votes(accumulator, rhos, thetas)
    
    # Draw lines on the image
    draw(line_img, rho, theta)

    #cv2.imshow('Lines', line_img)
    cv2.imwrite('lines-detected-{0}'.format(filename), line_img)
    #cv2.waitKey(0)
