import matplotlib.pyplot as plt
import data_x_new as s
import numpy as np
import random
np.set_printoptions(linewidth=400) # 列印陣列時每一行字的字元數上限
np.set_printoptions(threshold=np.inf)
menory=range(4,8)  # [4,5,6,7]
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

w=np.zeros([56,56])
for t in menory:         
    g=x[t,:].reshape(x[t,:].shape[0],1) # 56*1
    w+=np.multiply(x[t,:],g)
for i in range(56):
    w[i,i]=0



testing_data=np.zeros([8,56])
testing_data[:,:]=x[:,:]
ram=6
for i in range(8):
    a=random.sample(range(0,56), ram) # choose 6 numbers between 0 to 55
    for t in range(ram):
        testing_data[i,a[t]]=-1*testing_data[i,a[t]]
for f in menory:
    plt.figure(figsize=(3.5,5))
    for i in range(8):
       for y in range(7):
           if testing_data[f,(7*i)+y]==-1:
             plt.scatter(y+1,8-i,color='red',s=50,marker='s')  
    plt.axis([0,7.5,0,8.5])  



for i in menory:
    thr=0
    count=0
    testing_final=np.zeros(56)
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
     

        e=sum(np.absolute(testing_data[i,:]-testing_final[:]))

        testing_final[:]=testing_data[i,:]
        if e==0:
            thr=1
            
    for f in menory:
        thr_i=0
        thr_i=sum(np.absolute(testing_data[i,:]-x[f,:]))
        if thr_i==0:
            print('testing data'+str(i+1)+'perturbed by '+tit[f])
for f in menory:
    plt.figure(figsize=(3.5,5))
    for i in range(8):
        for y in range(7):
            if testing_data[f,(7*i)+y]==-1:
                plt.scatter(y+1,8-i,color='red',s=50,marker='s')  
    plt.axis([0,7.5,0,8.5])
            
