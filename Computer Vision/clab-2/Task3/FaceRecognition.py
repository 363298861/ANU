import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import math

def loadImg(files, num):
    faces = np.mat(np.zeros((num, 231 * 195)))
    j = 0
    for i in os.listdir(files):
        img = cv2.imread(files + i, 0)
        faces[j, :] = np.mat(img).flatten()
        j += 1
    return faces

def eigenface(faces):
    # Calculate the mean face from the data set
    mean = np.mean(faces, axis=1)
    # Normalize the train set
    trainset = faces - mean
    # Calculate the eigenvectors and eigenvalues
    S = trainset.T * trainset
    eigval, eigvec = np.linalg.eigh(np.mat(S))
    # 95% of the variance
    v = np.sum(eigval) * 0.95
    pc = 0
    # Sort the eigenvalues in descending order
    sortedInd = np.argsort(-eigval)
    sum = 0
    # Find how many eigenvalues consists 95% of variance
    for i in range(len(eigval)):
        sum += eigval[sortedInd[i]]
        if sum > v:
            pc = i + 1
            break
    print('The minimum number of principal component that contains 95% variation is {}'.format(pc))
    # Make k to be 12
    pc = 12
    eigfaces = dict()
    # Find the first k eigenvectors of covariance matrix
    eigvec = trainset * eigvec[:, sortedInd[0 : pc]]
    # reshape eigenfaces and plot them
    for i in range(pc):
        eigfaces[i] = np.reshape(eigvec[:, i], (231, 195))
        plt.imshow(eigfaces[i], 'gray')
        #plt.show()
    return mean, eigvec, trainset

def recognition(testset, mean, eigvec, trainset):
    # Normalize the test image
    test = testset - mean
    # Find the test image weights
    weight = eigvec.T * test
    # Calculate the training images weights
    weighttrain = eigvec.T * trainset
    # Calculate ||W - Wm|| and find the minimum
    dist = np.linalg.norm(weight - weighttrain, axis=0)
    sortarg = np.argsort(dist)
    # Return three smallest index
    return [i + 1 for i in sortarg[: 3]]

    # res = 0
    # resval = math.inf
    # for i in range(trainset.shape[1]):
    #     weightorig = eigvec.T * trainset[:, i]
    #     if np.linalg.norm(weight - weightorig) < resval:
    #         res = i
    #         resval = np.linalg.norm(weight - weightorig)
    # return res + 1

def locateFile(trainset, testset):
    j = 0
    for i in os.listdir('Yale-FaceA/trainingset/'):
        j += 1
        if j == trainset:
            print('The most similar image is {}'.format(i))
    k = 0
    for i in os.listdir('Yale-FaceA/myfaces/'):
        k += 1
        if k == testset:
            print('The input image is {}'.format(i))

faces = loadImg('Yale-FaceA/trainingset/', 144).T
# mean = np.mean(faces, axis=1)
# meanFace = np.reshape(mean, (231, 195))
# plt.imshow(meanFace, 'gray')
# plt.show()

tmp = np.reshape(faces[:, 0], (231, 195))
mean, eigvec, trainset = eigenface(faces)
testset = loadImg('Yale-FaceA/myfaces/', 1).T

result = recognition(testset[:, 0], mean, eigvec, trainset)
for r in result:
    locateFile(r, 1)

