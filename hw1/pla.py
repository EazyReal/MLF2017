import numpy as np
import sys
import random

filename = "./hw1_15_train.dat.txt"

with open(filename) as f:
    dataset = f.readlines()

#preprocess => data = [ [features(list), lab(int)] ]
dataset = [line.strip().rsplit(None, 1) for line in dataset]
dataset = [ [[1.0]+list(map(float, line[0].split())), int(line[1])] for line in dataset]
dataset = np.array(dataset)
#print(data[0])

#setting
DIMENSION = 5
EPOCH = 2000
LR = 0.5

def sign(x):
    tmp = np.sign(x)
    if tmp :
        return tmp
    else :
        return -1

def pla(dataset, key):
    w = np.array([0.0]*DIMENSION)
    count = 0
    #sign(0) = -1

    while True:
        flag = 0

        for i in key:
            data = dataset[i]
            s = np.dot(w, data[0])
            if sign(s) != data[1] :
                flag += 1
                count += 1
                w += np.multiply(data[1]*LR,data[0])

        if flag == 0 :
            break

    return w, count

if __name__ == "__main__":
    key = list(range(len(dataset)))

    update_sum = 0.0

    for i in range(EPOCH):
        random.shuffle(key)
        w, update = pla(dataset, key)
        update_sum += update

    print("update count = %f" %(update_sum/EPOCH))



"""
For Questions 15-20, you will play with PLA and pocket algorithm. First, we use an artificial data set to study PLA. The data set is in

https://www.csie.ntu.edu.tw/~htlin/mooc/datasets/mlfound_math/hw1_15_train.dat

Each line of the data set contains one (xn,yn) with xn∈R4. The first 4 numbers of the line contains the components of xn orderly, the last number is yn.

Please initialize your algorithm with w=0 and take sign(0) as −1.

Implement a version of PLA by visiting examples in the naive cycle using the order of examples in the data set. Run the algorithm on the data set. What is the number of updates before the algorithm halts?
"""