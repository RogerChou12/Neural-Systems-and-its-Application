# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import random
import math
from mpl_toolkits.mplot3d.axes3d import Axes3D

def cpn_output (inputs, w_1, w_2, N):
    distance=np.zeros(N)
    for neuron in range(N):
        distance[neuron]=0
        for s in range(2):
            distance[neuron]+=np.power((inputs[s]-w_1[neuron,s]), 2)     
        distance[neuron]=math.sqrt(distance[neuron]) # Compute the distance between inputs and neurons
        
    min_index=np.argmin(distance) # Winner neuron
    return w_2[min_index] # Returns the corresponding Grossberg weight

# Training and testing data
data_i=np.zeros([200, 2])
data_f=np.zeros(200)
for i in range(200):
    data_i[i, 0]=random.uniform(-1*(math.pi),math.pi)
    data_i[i, 1]=random.uniform(-1*(math.pi),math.pi)
    data_f[i]=data_i[i, 1]*math.sinh(data_i[i, 0])+data_i[i, 1]*math.cosh(data_i[i, 0])

# Plot training data and testing data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.title('Data')
ax.scatter(data_i[range(0,150), 0], data_i[range(0,150), 1], data_f[range(0, 150)], color='red', s=8, label='train')
ax.scatter(data_i[range(150,200), 0], data_i[range(150,200), 1], data_f[range(150, 200)], color='blue', s=8, label='test')
ax.legend()
plt.show()

N=10 # numbers of hidden neurons
w11=np.zeros([N, 2]) # weights of hidden layer (Kohonen layer) when sequence of input is fixed
w12=np.zeros([N, 2]) # weights of hidden layer when sequence of input is random
w21=np.zeros(N) # weights of Grossberg layer(output layer) when sequence of input is fixed
w22=np.zeros(N) # weights of Grossberg layer when sequence of input is random
for i in range(N):
    w11[i, 0]=w12[i, 0]=random.uniform(-1*(math.pi), math.pi)
    w11[i, 1]=w12[i, 0]=random.uniform(-1*(math.pi), math.pi)
    w21[i]=w22[i]=random.uniform(-40, 40)

# Plot initial weights of Grossberg layer
print('Initail  weights of Grossberg layer')
print(w21)
plt.figure()
plt.title('Initial Weights')
plt.scatter(data_i[range(0,150),0], data_i[range(0,150),1], color='red', s=8, label='train')
plt.scatter(w11[:,0], w11[:,1], color='blue', s=15, label='weight')
plt.legend(loc='upper right')
plt.show()

LR1_i=0.5 # Initial learning rate
LR2_i=0.5
mse=0
epoch=0
distance=np.zeros(N)

while epoch <60: # use fixed sequences of input data
    LR1=LR1_i*math.exp(-1*(epoch/100))
    LR2=LR2_i*math.exp(-1*(epoch/100))
    mse_i=0

    for k in range(150):
        for neuron in range(N):
            distance[neuron]=0
            for s in range(2):
                distance[neuron]+=np.power((data_i[k,s]-w11[neuron,s]), 2)     
            distance[neuron]=math.sqrt(distance[neuron]) # Distance between inputs and neurons
        min_index=np.argmin(distance) # The smallest neurons
        w21[min_index]+=LR2*(data_f[k]-w21[min_index]) # Update weights of Grossberg layer
        w11[min_index,:]+=LR1*(data_i[k,:]-w11[min_index,:]) # Update weights of hidden layer (Kohonen layer)
        mse_i+=np.power((cpn_output(data_i[k,:],w11,w21,N)-data_f[k]), 2) # Sum of MSE of outputs and targets

    mse=math.sqrt(mse_i)/150 # MSE per epoch
    epoch+=1

epoch=0
while epoch <60: # use random sequences of input data
    LR1=LR1_i*math.exp(-1*(epoch/100))
    LR2=LR2_i*math.exp(-1*(epoch/100))
    mse_i=0
    a=random.sample(range(0,150), 150) # sequence of input data is random
    data_i[range(0,150),:]=data_i[a,:]
    data_f[range(0,150)]=data_f[a]
    
    for k in range(150):
        for neuron in range(N):
            distance[neuron]=0
            for s in range(2):
                distance[neuron]+=np.power((data_i[k,s]-w12[neuron,s]), 2)     
            distance[neuron]=math.sqrt(distance[neuron])
        min_index=np.argmin(distance)
        w22[min_index]+=LR2*(data_f[k]-w22[min_index])
        w12[min_index,:]+=LR1*(data_i[k,:]-w12[min_index,:])
        mse_i+=np.power((cpn_output(data_i[k,:],w12,w22,N)-data_f[k]), 2) # Compute errors between outputs and predictions

    mse=math.sqrt(mse_i)/150 # Compute MSE between outputs and predictions
    epoch+=1

# Plot training data and final weights of Kohonen layer
print('Final weights of Kohonen layer(fixed)')
print(w21)
plt.figure()
plt.title('Sequence of input is fixed')
plt.scatter(data_i[:,0], data_i[:,1], color='red', s=5, label='train')
plt.scatter(w11[:,0], w11[:,1], color='blue', s=15, label='weights')
plt.legend(loc='upper right')
plt.show()

print('Final weights of Grossberg layer(random)')
print(w22)
# Plot training data and final weights of Grossberg layer
plt.figure()
plt.title('Sequence of input is random')
plt.scatter(data_i[:,0], data_i[:,1], color='red', s=5, label='train')
plt.scatter(w12[:,0],w12[:,1], color='blue', s=15, label='weights')
plt.legend(loc='upper right')
plt.show()

# testing
test_f1=np.zeros(50)
test_f2=np.zeros(50)
for i in range(150,200): 
    test_f1[i-150]=cpn_output(data_i[i,:], w11, w21, N)
    test_f2[i-150]=cpn_output(data_i[i,:], w12, w22, N)

# Plot testing data, weights, outputs
fig1 = plt.figure()
ap = fig1.add_subplot(111, projection='3d')
plt.title('Result(fixed sequence)')
ap.scatter(data_i[range(150,200),0],data_i[range(150,200),1],data_f[range(150,200)],s=15,marker='o',facecolor=(0,0,0,0), edgecolor='black',label='original')
ap.scatter(w11[:,0], w11[:,1], w21[:], color='red', s=15,marker='s', label='weights')
ap.scatter(data_i[range(150,200),0],data_i[range(150,200),1],test_f1[:], color='green', s=15, marker='x', label='test')
ap.legend()
plt.show()

# Plot testing data, weights, outputs
fig2 = plt.figure()
ap2 = fig2.add_subplot(111, projection='3d')
plt.title('Result(random sequence)')
ap2.scatter(data_i[range(150,200),0], data_i[range(150,200),1], data_f[range(150,200)], s=15, marker='o', facecolor=(0,0,0,0), edgecolor='black', label='original')
ap2.scatter(w12[:,0], w12[:,1], w22[:], color='red',s=15,marker='s',label='weights')
ap2.scatter(data_i[range(150,200),0], data_i[range(150,200),1], test_f2[:], color='green', s=15,marker='x', label='test')
ap2.legend()
plt.show()
