
import numpy as np
import matplotlib.pyplot  as plt

def costFunc(X, w, Y):
    predicted = np.dot(X,w)
    cost = np.square(predicted - Y)
    return np.sum(cost)/len(X)

def gradDesc(x, weights, y):
    n = len(x)
    predicted = np.dot(x,weights)
    cost = predicted - y
    gradient = np.dot(x.T,cost)/n
    return gradient


def readDataFile():
    fileName = 'Cereal_Train.csv'
    print("fileName: ", fileName)
    raw_data = open(fileName, 'rt')
    data = np.loadtxt(raw_data, usecols = (2,3,4,5,6,7,8,9,10,11,12,13,14), skiprows = 1, delimiter=",")
    return data

data = readDataFile()
print(data)
x = data[:,0:12]
y =data[:,12:13]
print("x", x)
print("y", y)
means = np.mean(x, axis = 0)
print("mean:", means)
stdev = np.std(x,axis = 0)
print("standard deviation: ", stdev)
x = (x-means)/stdev

x1 = np.ones((len(x),1))
x = np.hstack((x1,x))
print(x)

weights = np.zeros((len(x[0,:]),1))
LR = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 1.3]
maxIter = 100

iterations = 0
minDiff = 0.0001
vectDiff = 1
print("initial cost", costFunc(x,weights,y))
for lr in LR:
    weights = np.zeros([x.shape[1],1])
    vectDiff = 1
    iterations = 0
    print(lr)
    while iterations<maxIter and vectDiff > minDiff:
        gradients = gradDesc(x,weights,y)
        weights -= lr*gradients
        vectDiff = np.linalg.norm(gradients)
        iterations += 1
    print("cost", costFunc(x,weights,y))
    print("weights:", weights)






