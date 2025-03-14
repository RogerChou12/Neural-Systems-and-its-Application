# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import random
import math
from mpl_toolkits.mplot3d.axes3d import Axes3D

def cpn_output (input,w_1,w_2,N):
    distance=np.zeros(N)
    for neuron in range(N):
        distance[neuron]=0
        for s in range(2):
            distance[neuron]+=np.power((input[s]-w_1[neuron,s]),2)     
        distance[neuron]=math.sqrt(distance[neuron]) # Compute the distance between inputs and neurons
        
    min_index=np.argmin(distance) # Winner neuron

    return w_2[min_index] # Returns the corresponding Grossberg weight

data_i=np.zeros([200,2])
data_f=np.zeros(200)
for i in range(200): # Generate training data randomly
    data_i[i,0]=random.uniform(-1*(math.pi),math.pi)
    data_i[i,1]=random.uniform(-1*(math.pi),math.pi)
    data_f[i]=data_i[i,1]*math.sinh(data_i[i,0])+data_i[i,1]*math.cosh(data_i[i,0])

# Plot training data and testing data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.title('DATA')
ax.scatter(data_i[range(0,150),0],data_i[range(0,150),1],data_f[range(0,150)], color='red',s=8,label='train')
ax.scatter(data_i[range(150,200),0],data_i[range(150,200),1],data_f[range(150,200)], color='blue',s=8,label='test')
plt.legend()
plt.show()

# Initial weights and number of neuron is 1
N=1
w1=np.array([data_i[0,:]])
w2=np.array([data_f[0]])

# Start training
th1=0.5 # Distance threshold
th2=0.5
LR1_i=0.5 # Initial learning rate
LR2_i=0.5
epoch=0
mse_t=np.zeros(60)
while epoch <60:
    LR1=LR1_i*math.exp(-1*(epoch/1000))
    LR2=LR2_i*math.exp(-1*(epoch/1000))
    mse_i=0
    samples=random.sample(range(0,150), 150) # Choose 150 training data randomly
    data_i[range(0,150),:]=data_i[samples,:]
    data_f[range(0,150)]=data_f[samples]

    for k in range(150):
        distance=np.zeros(N)
        for neuron in range(N):
            distance[neuron]=0
            for s in range(2):
                distance[neuron]+=np.power((data_i[k,s]-w1[neuron,s]),2)     
            distance[neuron]=math.sqrt(distance[neuron]) # Compute distance between inputs and neurons
        min_index=np.argmin(distance) # The winner neuron
        # If error and distance are within thresholds, update the winner neuron
        if abs(data_f[k]-w2[min_index])<th2 and distance[min_index]<th1:
            w2[min_index]+=LR2*(data_f[k]-w2[min_index]) # Update weights of Grossberg layer
            w1[min_index,:]+=LR1*(data_i[k,:]-w1[min_index,:]) # Update weights of Kohonen layer
        else:  # create a new neuron
            w1=np.append(w1,[data_i[k,:]],axis=0) # Append a row to 'w1'
            w2=np.append(w2,data_f[k])
            N+=1
        
        mse_i+=np.power((cpn_output(data_i[k,:],w1,w2,N)-data_f[k]),2) # Sum of every MSE of output and targets

    mse_t[epoch]=math.sqrt(mse_i)/150 # MSE per epoch
    epoch+=1

# Plot training data and weights of Kohonen layer
plt.figure()
plt.title('WEIGHTS')
plt.scatter(data_i[range(0,150),0],data_i[range(0,150),1], color='red',s=5,label='train')
plt.scatter(w1[:,0],w1[:,1], color='blue',s=15,label='weights')
plt.legend(loc='upper right')
plt.show()

# Start testing
test_f=np.zeros(50)
error = np.zeros(50)
for i in range(150,200): 
    test_f[i-150]=cpn_output(data_i[i,:],w1,w2,N)
    error[i-150] = abs(data_f[i]-test_f[i-150]) # Difference between testing output and original output

print('Final number of hidden neurons = '+str(N))
# Plot testing data and outputs
fig1 = plt.figure()
ap = fig1.add_subplot(111, projection='3d')
plt.title('RESULTS')
ap.scatter(data_i[range(150,200),0],data_i[range(150,200),1],data_f[range(150,200)],s=15,marker='o',label='original',facecolor=(0,0,0,0), edgecolor='black')
ap.scatter(data_i[range(150,200),0],data_i[range(150,200),1],test_f[:], color='green',s=15,marker='x',label='result')
plt.legend()
plt.show()

# Plot testing data, outputs, error
plt.figure()
plt.title('ERROR')
plt.plot(range(150,200),error[:],color='red',label='error')
plt.plot(range(150,200),data_f[range(150,200)],color='blue',label='data')
plt.plot(range(150,200),test_f[:],color='green',label='CPN')
plt.legend(loc='upper right')
plt.show()   
    
