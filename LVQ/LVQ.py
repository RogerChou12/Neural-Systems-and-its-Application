# -*- coding: utf-8 -*-

import numpy as np
import math

x = np.array([[1,0,0,0,0],[2,0,0,0,0],[3,0,0,0,0],[4,0,0,0,0],[5,0,0,0,0],
             [3,1,0,0,0],[3,2,0,0,0],[3,3,0,0,0],[3,4,0,0,0],[3,5,0,0,0],
             [3,3,1,0,0],[3,3,2,0,0],[3,3,3,0,0],[3,3,4,0,0],[3,3,5,0,0],[3,3,6,0,0],[3,3,7,0,0],[3,3,8,0,0],
             [3,3,3,1,0],[3,3,3,2,0],[3,3,3,3,0],[3,3,3,4,0],
             [3,3,6,1,0],[3,3,6,2,0],[3,3,6,3,0],[3,3,6,4,0],
             [3,3,6,2,1],[3,3,6,2,2],[3,3,6,2,3],[3,3,6,2,4],[3,3,6,2,5],[3,3,6,2,6]])
Xin = x.transpose()
w = np.array([[1,0,0,0,0],
             [3,1,0,0,0],
             [3,3,1,0,0],
             [3,3,3,1,0],
             [3,3,6,1,0],
             [3,3,6,2,1]])  # initial weights
w1 = w.transpose()
w1=w1.astype(float) # Convert weights into float format for precision
print('initial weights=')
print(w1)
M = 6   # 6 classes
y = np.zeros([M,1])
lr0 = 0.01
cx = np.zeros([1,32]) # Class labels for training data
for i in range(32):   # Divides the training data into 6 predefined classes
    # Assigns each training sample to a class
    if i<5:
        cx[0,i] = 1
    if i>4 and i<10:
        cx[0,i] = 2
    if i>9 and i<18:
        cx[0,i] = 3
    if i>17 and i<22:
        cx[0,i] = 4
    if i>21 and i<26:
        cx[0,i] = 5
    if i>25:
        cx[0,i] = 6
cw = np.array([1,2,3,4,5,6])  # outputs: 6 classes

# training
for e in range(2000):
    lr = lr0*math.exp(-1*(e/1000)) # Learning rate decays exponentially
    for d in range(32):
        dis = np.zeros([6,1])
        win = 0
        win_index = 0
        for i in range(M):
            # Compute Euclidean distance
            dif = np.zeros([5,1])
            dif = np.power((Xin[:,d] - w1[:,i]),2)
            dis[i] = math.sqrt(np.sum(dif))
        win = min(dis)
        win_index = np.argmin(dis) # Finds the neuron with the smallest distance (Winner Neuron)
        new = np.zeros([5,1])
        if cw[win_index] == cx[0,d]: # If class matches, strengthen weights
            new = w1[:,win_index] + lr*(Xin[:,d]-w1[:,win_index])
            w1[:,win_index] = new
        if cw[win_index] != cx[0,d]: # If class does not match, weaken weights
            new = w1[:,win_index] - lr*(Xin[:,d]-w1[:,win_index])
            w1[:,win_index] = new
print('final weights=')
print(w1)

# testing data
test = np.array([[6,0,0,0,0],
                 [3,6,0,0,0],
                 [3,3,9,0,0],
                 [3,3,3,5,0],
                 [3,3,6,5,0],
                 [3,3,6,2,7]])
test1=test.transpose()
# start testing
result = np.zeros([6,6]) # Store distances for classification
# Compute Euclidean distance
for j in range(6): # 6 testing samples
    for i in range(6): # 6 classes
        test_dif = np.zeros([5,1])
        test_dif = np.power((test1[:,j] - w1[:,i]),2)
        result[j,i] = math.sqrt(np.sum(test_dif))

win_c = np.zeros([6,1])  # output category
win_c = win_c.astype(int)
win_value = np.zeros([6,1])  # the value of the output category
print('Results of testing')
for i in range(6):  # find the correct category
    win_value[i] = min(result[i,:]) # the smallest distance between testing samples and classes
    win_c[i] = np.argmin(result[i,:])
    print(str(test[i,:])+' -> category='+str(cw[win_c[i]]))
