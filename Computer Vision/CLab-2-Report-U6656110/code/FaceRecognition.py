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
    mean = np.mean(faces, axis=1)
    trainset = faces - mean
    S = trainset.T * trainset
    eigval, eigvec = np.linalg.eigh(np.mat(S))
    v = np.sum(eigval) * 0.95
    pc = 0
    sortedInd = np.argsort(-eigval)
    sum = 0
    for i in range(len(eigval)):
        sum += eigval[sortedInd[i]]
        if sum > v:
            pc = i + 1
            break
    print('The minimum number of principal component that contains 95% variation is {}'.format(pc))
    pc = 12
    eigfaces = dict()
    eigvec = trainset * eigvec[:, sortedInd[0 : pc]]
    for i in range(pc):
        eigfaces[i] = np.reshape(eigvec[:, i], (231, 195))
        # plt.imshow(eigfaces[i], 'gray')
        # plt.show()
    return mean, eigvec, trainset

def recognition(testset, mean, eigvec, trainset):
    test = testset - mean
    weight = eigvec.T * test
    weighttrain = eigvec.T * trainset
    dist = np.linalg.norm(weight - weighttrain, axis=0)
    sortarg = np.argsort(dist)
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
    for i in os.listdir('Yale-FaceA/testset/'):
        k += 1
        if k == testset:
            print('The input image is {}'.format(i))

# mean = np.mean(faces, axis=1)
# meanFace = np.reshape(mean, (231, 195))
# plt.imshow(meanFace, 'gray')
faces = loadImg('Yale-FaceA/trainingset/', 135).T
tmp = np.reshape(faces[:, 0], (231, 195))
mean, eigvec, trainset = eigenface(faces)
testset = loadImg('Yale-FaceA/testset/', 10).T
result = recognition(testset[:, 1], mean, eigvec, trainset)
for r in result:
    locateFile(r, 2)

