import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

def my_kmeans(img, k):
    row, col, clr = img.shape
    # initialize centroids randomly
    centroids = {
        i: img[np.random.randint(0, row), np.random.randint(0, col)]
        for i in range(k)
    }
    # Initialize centroids using kmeans++
    centroids = kmeansplus(img, k)
    changed = True
    res = None
    # Iterate until centroids don't change
    while changed:
        changed = False
        print(centroids)
        # assign all the points to its nearest centroid
        res, mean = assign(img, centroids)
        # Check whether centroids changed
        for i in range(k):
            cent = mean[i] / (len(res[i]) + 1)
            # Calculate the norm of previous centroids and new centroids
            # The results less than 1 is acceptable
            if np.linalg.norm(np.asarray(centroids[i]) - np.asarray(cent)) > 0.1:
                centroids[i] = cent
                changed = True
    return centroids, res



def assign(img, centroids):
    row, col, clr = img.shape
    cluster = len(centroids)
    res = {
        i: []
        for i in range(cluster)
    }
    mean = {
        i: np.zeros((clr, ))
        for i in range(cluster)
    }
    # Iterate through all pixels in the image
    # assign it to its nearest centroid cluster
    for i in range(row):
        for j in range(col):
            minimum = 0
            idx = 0
            for k in centroids.keys():
                dist = np.linalg.norm(img[i, j] - centroids[k])
                if minimum == 0 or dist < minimum:
                    minimum = dist
                    idx = k
            res[idx].append((i, j))
            mean[idx] += img[i, j]
    return res, mean

def kmeansplus(img, k):
    row, col, clr = img.shape
    # Initialize the first centroid randomly
    first = img[np.random.randint(0, row), np.random.randint(0, col)]
    centroids = dict()
    centroids[0] = first
    dist = np.zeros((row, col))
    # Find the remaining k - 1 centroids
    for _ in range(k - 1):
        total = 0.0
        for i in range(row):
            for j in range(col):
                # For a pixel in the image, find its nearest centroid
                # Then record the distance and sum them
                dist[i, j] = get_closest_dist(img[i, j], centroids)
                total += dist[i, j]
        # We multiply the total distance by a random number between 0 ~ 1
        # Then iterate through data points to find which pixel locates there
        # This pixel is chosen as the next centroid
        total *= np.random.random()
        centers = len(centroids)
        for m in range(row):
            for n in range(col):
                total -= dist[m, n]
                if total > 0:
                    continue
                centroids[_ + 1] = img[m, n]
                break
            if centers == len(centroids) + 1:
                break
    return centroids


def get_closest_dist(pixel, centroids):
    min = math.inf
    for i in centroids.keys():
        dist = np.linalg.norm(centroids[i] - pixel)
        if dist < min:
            min = dist
    return min


imgOrig = plt.imread('peppers.png')
imgProc = cv2.cvtColor(imgOrig, cv2.COLOR_RGB2Lab)
r, c, ch = imgOrig.shape
coordinates = np.zeros((r, c, 2))
for i in range(r):
    for j in range(c):
        coordinates[i, j] = [i, j]
imgCoor = np.dstack((imgProc, coordinates))
#print(imgCoor[100, 110])
centroid, res = my_kmeans(imgProc, 4)
img = np.zeros((imgOrig.shape))
color = [[106, 125, 142], [193, 210, 214], [176, 102, 96], [219, 173, 114], [94, 119, 3], [155, 175, 142], [249, 211, 165], [234, 196, 184], [202, 143, 66]]
for i in res.keys():
    for x, y in res[i]:
        img[x, y] = color[i]

plt.imshow(img.astype(np.uint8))
plt.show()

# img = cv2.imread('peppers.png')
# Z = img.reshape((-1,3))
# # convert to np.float32
# Z = np.float32(Z)
# # define criteria, number of clusters(K) and apply kmeans()
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# K = 5
# ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
# # Now convert back into uint8, and make original image
# center = np.uint8(center)
# res = center[label.flatten()]
# res2 = res.reshape((img.shape))
# plt.imshow(res2)
# plt.show()