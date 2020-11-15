import numpy as np
import cv2
import math
from matplotlib import pyplot as plt

def hough_circle(img, threshold=100):
    # Initialize the accumulator
    row, col = img.shape
    thetas = np.deg2rad(np.arange(360))
    width = min(row, col)
    rad = np.arange(5, 6)
    acc = np.zeros((row, col, len(rad)), dtype=int)
    # Cache reusable data
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    # Extract edge points
    row_idx, col_idx = np.nonzero(img)
    for r in range(len(rad)):
        print('Detecting circles of radius {}'.format(rad[r]))
        for i in range(len(row_idx)):
            x = col_idx[i]
            y = row_idx[i]
            for t in range(len(thetas)):
                a = int(y - rad[r] * sin_t[t])
                b = int(x - rad[r] * cos_t[t])
                if a >= 0 and a < row and b >= 0 and b < col:
                    acc[a, b, r] += 1
    indices = np.argwhere(acc > threshold)
    print(indices.shape)
    print(indices)
    return acc, indices, rad

def draw(img, acc, indices, rad, threshold=200):
    # Sort the accumulator by its indices
    idx = np.argsort(-acc, axis=None)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    row = img.shape[0]
    col = img.shape[1]
    rds = len(rad)
    # The grid size of the 3d array in order to find index
    grid = rds * col
    circles = list()
    # Iterate through all elements
    for i in idx:
        # Find the original indices
        cur = np.zeros(3, dtype=int)
        cur[0] = int(i // grid)
        cur[1] = int((i % grid) // rds)
        cur[2] = int((i % grid) % rds)
        # Compared with the threshold, if the accumulator is
        # less than the threshold, then stop because the array is sorted
        # all the elements after are less than threshold
        if acc[cur[0], cur[1], cur[2]] < threshold:
            break
        if len(circles) == 0:
            circles.append(cur)
        else:
            # Iterate through the results, if there is one result that are close
            # to an existing result, ignore it. If there isn't one, append to results.
            for c, cidx in enumerate(circles):
                if np.linalg.norm(cidx - cur) < 20:
                    break
                if c == len(circles) - 1:
                    circles.append(cur)
    # Draw all the results in the image
    for ro, co, rd in circles:
        print(acc[ro, co, rd])
        radius = int(rad[rd])
        cv2.circle(img, (co, ro), radius, (0, 0, 255), 2)





img = cv2.imread('f4.jpg', 0)
line_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
gauss = cv2.GaussianBlur(img, (5, 5), 1)
edge_img = cv2.Canny(gauss, 50, 150, None, 3)
acc, idx, rad = hough_circle(edge_img)
draw(line_img, acc, idx, rad)
cv2.imwrite('circle-detected-face6.jpg', line_img)
plt.imshow(line_img)
plt.show()