# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import random
import math
from mpl_toolkits.mplot3d.axes3d import Axes3D

def cpn (ij,w_1,w_2,N):
    #print(ij)
    d=np.zeros(N)
    for e in range(N):
        d[e]=0
        for s in range(2):
            d[e]+=np.power((ij[s]-w_1[e,s]),2)     
        d[e]=math.sqrt(d[e])
        
    min_index=np.argmin(d)

    return w_2[min_index]
data_i=np.zeros([200,2])
data_f=np.zeros(200)
for i in range(200):
    data_i[i,0]=random.uniform(-1*(math.pi),math.pi)
    data_i[i,1]=random.uniform(-1*(math.pi),math.pi)
    data_f[i]=data_i[i,1]*math.sinh(data_i[i,0])+data_i[i,1]*math.cosh(data_i[i,0])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.title('Data')
ax.scatter(data_i[range(0,150),0],data_i[range(0,150),1],data_f[range(0,150)], color='red',s=8,label='train')
ax.scatter(data_i[range(150,200),0],data_i[range(150,200),1],data_f[range(150,200)], color='blue',s=8,label='test')
ax.legend()
plt.show()

N=10;
w11=np.zeros([N,2]) # weights of hidden layer
w12=np.zeros([N,2]) # weights of hidden layer when sequence of input is random
w21=np.zeros(N) # weight of Grossberg layer(output layer)
w22=np.zeros(N)
for i in range(N):
    w11[i,0]=w12[i,0]=random.uniform(-1*(math.pi),math.pi)
    w11[i,1]=w12[i,0]=random.uniform(-1*(math.pi),math.pi)
    w21[i]=w22[i]=random.uniform(-40,40)
print('initail  weights of Grossberg layer')
print(w21)
plt.figure()
plt.title('Initial Weights')
plt.scatter(data_i[range(0,150),0],data_i[range(0,150),1], color='red',s=8,label='train')
plt.scatter(w11[:,0],w11[:,1], color='blue',s=15,label='weight')
plt.legend(loc='upper right')
plt.show()

LR1_i=0.5
LR2_i=0.5
mse=1
epoch=0
d=np.zeros(N)

while epoch <60: #sequence is fixed
    LR1=LR1_i*math.exp(-1*(epoch/100))
    LR2=LR2_i*math.exp(-1*(epoch/100))
    mse_i=0

    for k in range(150):
        for e in range(N):
            d[e]=0
            for s in range(2):
                d[e]+=np.power((data_i[k,s]-w11[e,s]),2)     
            d[e]=math.sqrt(d[e])
        min_index=np.argmin(d)
        w21[min_index]+=LR2*(data_f[k]-w21[min_index])
        w11[min_index,:]+=LR1*(data_i[k,:]-w11[min_index,:])
        #print(min_index)
        mse_i+=np.power((cpn(data_i[k,:],w11,w21,N)-data_f[k]),2)

    mse=math.sqrt(mse_i)/150

    epoch+=1
epoch=0
while epoch <60: #sequence is random
    LR1=LR1_i*math.exp(-1*(epoch/100))
    LR2=LR2_i*math.exp(-1*(epoch/100))
    mse_i=0
    a=random.sample(range(0,150), 150) # sequence of input data is random
    data_i[range(0,150),:]=data_i[a,:]
    data_f[range(0,150)]=data_f[a]
    
    for k in range(150):
        for e in range(N):
            d[e]=0
            for s in range(2):
                d[e]+=np.power((data_i[k,s]-w12[e,s]),2)     
            d[e]=math.sqrt(d[e])
        min_index=np.argmin(d)
        w22[min_index]+=LR2*(data_f[k]-w22[min_index])
        w12[min_index,:]+=LR1*(data_i[k,:]-w12[min_index,:])
                     
        #print(min_index)
        mse_i+=np.power((cpn(data_i[k,:],w12,w22,N)-data_f[k]),2)

    mse=math.sqrt(mse_i)/150

    epoch+=1

print('final weights of Grossberg layer(fixed)')
print(w21)
plt.figure()
plt.title('sequence of input is fixed')
plt.scatter(data_i[:,0],data_i[:,1], color='red',s=5,label='train')
plt.scatter(w11[:,0],w11[:,1], color='blue',s=15,label='weights')
plt.legend(loc='upper right')
plt.show()

print('final weights of Grossberg layer(random)')
print(w22)
plt.figure()
plt.title('sequence of input is random')
plt.scatter(data_i[:,0],data_i[:,1], color='red',s=5,label='train')
plt.scatter(w12[:,0],w12[:,1], color='blue',s=15,label='weights')
plt.legend(loc='upper right')
plt.show()

# testing
test_f1=np.zeros(50)
test_f2=np.zeros(50)
for i in range(150,200): 
    test_f1[i-150]=cpn(data_i[i,:],w11,w21,N)
    test_f2[i-150]=cpn(data_i[i,:],w12,w22,N)

#print(w2)
fi = plt.figure()
ap = fi.add_subplot(111, projection='3d')
plt.title('result(fixed sequence)')
ap.scatter(data_i[range(150,200),0],data_i[range(150,200),1],data_f[range(150,200)],s=15,marker='o',facecolor=(0,0,0,0), edgecolor='black',label='original')
ap.scatter(w11[:,0],w11[:,1],w21[:], color='red',s=15,marker='s',label='weights')
ap.scatter(data_i[range(150,200),0],data_i[range(150,200),1],test_f1[:], color='green',s=15,marker='x',label='test')
ap.legend()
plt.show()

fi2 = plt.figure()
ap2 = fi2.add_subplot(111, projection='3d')
plt.title('result(random sequence)')
ap2.scatter(data_i[range(150,200),0],data_i[range(150,200),1],data_f[range(150,200)],s=15,marker='o',facecolor=(0,0,0,0), edgecolor='black',label='original')
ap2.scatter(w12[:,0],w12[:,1],w22[:], color='red',s=15,marker='s',label='weights')
ap2.scatter(data_i[range(150,200),0],data_i[range(150,200),1],test_f2[:], color='green',s=15,marker='x',label='test')
ap2.legend()
plt.show()
