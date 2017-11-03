import numpy as np
import sys
import random

filename = "./hw1_18_train.dat.txt"
filename_test = "./hw1_18_test.dat.txt"

#preprocess => data = [ [features(list), lab(int)] ]
def preprocess(filename):
    with open(filename) as f:
        dataset = f.readlines()
    dataset = [line.strip().rsplit(None, 1) for line in dataset]
    dataset = [ [[1.0]+list(map(float, line[0].split())), int(line[1])] for line in dataset]
    dataset = np.array(dataset)
    return dataset

#print(data[0])

#setting
DIMENSION = 5
EPOCH = 2000
UPDATE = 100
LR = 1.0

def sign(x):
    tmp = np.sign(x)
    if tmp :
        return tmp
    else :
        return -1

def validate(w, dataset):
    count = 0.0
    for data in dataset:
        if sign(np.dot(w, data[0])) != data[1] : count += 1
    return count/len(dataset)

def pocket_pla(dataset, update):
    #sign(0) = -1
    #random.seed(TIME)
    w_cap = np.array([0.0]*DIMENSION)
    w = np.array([0.0]*DIMENSION)
    key = list(range(len(dataset)))

    for step in range(update):
        random.shuffle(key)
        data = dataset[0]
        for i in key:
            data = dataset[i]
            s = np.dot(w, data[0])
            if sign(s) != data[1] : break

        w = w + np.multiply(LR*data[1],data[0])

        if validate(w, train_D) < validate(w_cap, train_D) :
            w_cap = w 

    return w_cap, w

if __name__ == "__main__":

    train_D = preprocess(filename)
    test_D = preprocess(filename_test)

    error_sum_cap = 0.0
    error_sum = 0.0

    for i in range(EPOCH):
        w_cap, w = pocket_pla(train_D, UPDATE)
        error_sum_cap += validate(w_cap, test_D)
        error_sum += validate(w, test_D)

    print("avg error of cap = %f\n avg error of update = %f" %(error_sum_cap/EPOCH, error_sum/EPOCH))



"""
For Questions 15-20, you will play with PLA and pocket algorithm. First, we use an artificial data set to study PLA. The data set is in

https://www.csie.ntu.edu.tw/~htlin/mooc/datasets/mlfound_math/hw1_15_train.dat

Each line of the data set contains one (xn,yn) with xn∈R4. The first 4 numbers of the line contains the components of xn orderly, the last number is yn.

Please initialize your algorithm with w=0 and take sign(0) as −1.

Implement a version of PLA by visiting examples in the naive cycle using the order of examples in the data set. Run the algorithm on the data set. What is the number of updates before the algorithm halts?
"""