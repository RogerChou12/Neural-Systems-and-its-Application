import matplotlib.pyplot as plt
import data_x_new as s
import numpy as np
import random
np.set_printoptions(linewidth=400) # 列印陣列時每一行字的字元數上限
np.set_printoptions(threshold=np.inf)
memory=range(4,8)  # [4,5,6,7]
x=np.zeros([8,56])
tit=['五','亢','云','了','乙','乃','久','人']
for i in range(8):
    x[i,:]=s.data(i)
    for c in range(56):
        if x[i,c]==1:
            x[i,c]=-1
        elif x[i,c]==0:
            x[i,c]=1
"""
for f in range(8):
    plt.figure(figsize=(3.5,5))
    for i in range(8):
       for y in range(7):
           if x[f,(7*i)+y]==-1:
             plt.scatter(y+1,8-i,color='red',s=50,marker='s')  
    plt.axis([0,7.5,0,8.5]) 
"""

w=np.zeros([56,56]) # Initialize 56×56 weight matrix
for t in memory:         
    g=x[t,:].reshape(x[t,:].shape[0],1) # Convert row to column (56*1)
    w+=np.multiply(x[t,:],g) # compute weights
for i in range(56):
    w[i,i]=0 # Prevents neurons from self-reinforcing

testing_data=np.zeros([8,56])
testing_data[:,:]=x[:,:]
noise=6 # Number of bits to flip
for i in range(8):
    flipped=random.sample(range(0,56), noise) # choose 6 numbers between 0 to 55 to be the noise
    for t in range(noise):
        testing_data[i,flipped[t]]=-1*testing_data[i,flipped[t]] # Flip selected bits
for f in memory:
    plt.figure(figsize=(3.5,5))
    for i in range(8):
       for y in range(7):
           if testing_data[f,(7*i)+y]==-1:
             plt.scatter(y+1,8-i,color='red',s=50,marker='s')  
    plt.axis([0,7.5,0,8.5])  

for i in memory:
    thr=0 # Convergence flag
    count=0
    testing_final=np.zeros(56) # Store the final stabilized state
    while thr==0 and count!=20:
        count+=1
        x_til=np.zeros(56)
        
        x_til[:]=w[:,:].dot(testing_data[i,:])
 
        
        for t in range(56):
            if x_til[t]>0:
                testing_data[i,t]=1
            elif x_til[t]<0:
                testing_data[i,t]=-1
            elif x_til[t]==0:
                testing_data[i,t]=testing_data[i,t]

        error=sum(np.absolute(testing_data[i,:]-testing_final[:])) # check if stable

        testing_final[:]=testing_data[i,:]
        if error==0:
            thr=1 # Stop if no changes
            
    for f in memory:
        thr_i=0
        thr_i=sum(np.absolute(testing_data[i,:]-x[f,:])) # Compare with original
        if thr_i==0:
            print('testing data'+str(i+1)+'perturbed by '+tit[f])
for f in memory:
    plt.figure(figsize=(3.5,5))
    for i in range(8):
        for y in range(7):
            if testing_data[f,(7*i)+y]==-1:
                plt.scatter(y+1,8-i,color='red',s=50,marker='s')  
    plt.axis([0,7.5,0,8.5])
            
