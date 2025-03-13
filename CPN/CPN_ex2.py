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
for i in range(200):
    data_i[i,0]=random.uniform(-1*(math.pi),math.pi)
    data_i[i,1]=random.uniform(-1*(math.pi),math.pi)
    data_f[i]=data_i[i,1]*math.sinh(data_i[i,0])+data_i[i,1]*math.cosh(data_i[i,0])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.title('training and testing data')
ax.scatter(data_i[range(0,150),0],data_i[range(0,150),1],data_f[range(0,150)], color='red',s=8,label='train')
ax.scatter(data_i[range(150,200),0],data_i[range(150,200),1],data_f[range(150,200)], color='blue',s=8,label='test')
plt.legend()
plt.show()

def cpn (N):
    w1=np.zeros([N,2])
    w2=np.zeros(N)
    for i in range(N):
        w1[i,0]=random.uniform(-1*(math.pi),math.pi)
        w1[i,1]=random.uniform(-1*(math.pi),math.pi)
        w2[i]=random.uniform(-40,40)
    
    plt.figure()
    plt.title('initial weights of hidden layer N='+str(N))
    plt.scatter(data_i[:,0],data_i[:,1], color='red',s=5,label='data')
    plt.scatter(w1[:,0],w1[:,1], color='blue',s=15,label='hidden layer')
    plt.legend(loc='upper right')
    plt.show()
    
    print('inital weights of Grossberg layer N='+str(N))
    print(w2)
    
    # start training
    LR1_i=0.5
    LR2_i=0.5
    mse=1
    epoch=0
    d=np.zeros(N)
    mse_t=np.zeros(100000)
    while epoch <60:
        LR1=LR1_i*math.exp(-1*(epoch/100))
        LR2=LR2_i*math.exp(-1*(epoch/100))
        mse_i=0
        a=random.sample(range(0,150), 150)
        data_i[range(0,150),:]=data_i[a,:]
        data_f[range(0,150)]=data_f[a]
        
        for k in range(150):
            for e in range(N):
                d[e]=0
                for s in range(2):
                    d[e]+=np.power((data_i[k,s]-w1[e,s]),2)     
                d[e]=math.sqrt(d[e])
            min_index=np.argmin(d)
            w2[min_index]+=LR2*(data_f[k]-w2[min_index])
            w1[min_index,:]+=LR1*(data_i[k,:]-w1[min_index,:])
                         
            #print(min_index)
            mse_i+=np.power((cpn_output(data_i[k,:],w1,w2,N)-data_f[k]),2)
    
        mse=math.sqrt(mse_i)/150
        mse_t[epoch]=mse 
        epoch+=1
    
    print('MSE N='+str(N))
    print(mse_t)
    print('final weights of Grossberg layer N='+str(N))
    print(w2)
    
    plt.figure()
    plt.title('final weights of hidden layer N='+str(N))
    plt.scatter(data_i[:,0],data_i[:,1], color='red',s=5,label='data')
    plt.scatter(w1[:,0],w1[:,1], color='blue',s=15,label='hidden layer')
    plt.legend(loc='upper right')
    plt.show()

    # start testing
    test_f=np.zeros(50)
    err_f=np.zeros(50)
    for i in range(150,200): 
        test_f[i-150]=cpn_output(data_i[i,:],w1,w2,N)
        err_f[i-150]=abs(data_f[i]-test_f[i-150]) # difference between testing output and original output
    
    #print(w2)
    fi = plt.figure()
    ap = fi.add_subplot(111, projection='3d')
    plt.title('Result N='+str(N))
    ap.scatter(data_i[range(150,200),0],data_i[range(150,200),1],data_f[range(150,200)],s=15,marker='o',label='original',facecolor=(0,0,0,0), edgecolor='black')
    ap.scatter(w1[:,0],w1[:,1],w2[:], color='red',s=15,marker='s',label='weights')
    ap.scatter(data_i[range(150,200),0],data_i[range(150,200),1],test_f[:], color='green',s=15,marker='x',label='CPN')
    plt.legend()
    plt.show()
    
    fc = plt.figure()
    ap1 = fc.add_subplot(111, projection='3d')
    plt.title('Difference N='+str(N))
    ap1.scatter(data_i[range(150,200),0],data_i[range(150,200),1],err_f[:], color='green',s=15,marker='x')
    plt.show()
    
    plt.figure()
    plt.title('curve')
    plt.plot(range(150,200),data_f[range(150,200)],color='blue',label='DATA')
    plt.plot(range(150,200),test_f[:],color='green',label='CPN')
    plt.plot(range(150,200),err_f[:],color='red',label='ERROR')
    plt.legend()
    plt.show()
    return 0

cpn(5) # Neurons=5
cpn(10) # Neurons=10