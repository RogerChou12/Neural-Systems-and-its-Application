# 隨機選取4筆資料，測試BAM清洗效果

import matplotlib.pyplot as plt
import data_x_new as s
import numpy as np
import random
np.set_printoptions(linewidth=400)
np.set_printoptions(threshold=np.inf)
menory=random.sample(range(0,8),4) # choose 4 data randomly
x=np.zeros([8,56])
tit=['五','亢','云','了','乙','乃','久','人']
for i in range(8):
    x[i,:]=s.data(i)
    for c in range(56):
        if x[i,c]==1:
            x[i,c]=-1
        elif x[i,c]==0:
            x[i,c]=1

m=28
w=np.zeros([56-m,m])
for t in menory:         
    g=x[t,range(m,56)].reshape(x[t,range(m,56)].shape[0],1)
    w+=np.multiply(x[t,range(0,m)],g)

#print(w)

testing_data=np.zeros([8,56])
testing_data[:,:]=x[:,:]
ram=6
for i in range(8):
    a=random.sample(range(0,56), ram)
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
    xy=0
    while thr==0 and count!=20:
        count+=1
        x_til=np.zeros(m)
        y_til=np.zeros(56-m)

        if xy==0:
            
            y_til[:]=w[:,:].dot(testing_data[i,range(0,m)])
            for t in range(m,56):
                if y_til[t-m]>0:
                    testing_data[i,t]=1
                elif y_til[t-m]<0:
                    testing_data[i,t]=-1
                elif y_til[t-m]==0:
                    testing_data[i,t]=testing_data[i,t]

            xy=1
        elif xy!=0:
            x_til[:]=(w[:,:].transpose()).dot(testing_data[i,range(m,56)])
            for t in range(m):
                if x_til[t]>0:
                    testing_data[i,t]=1
                elif x_til[t]<0:
                    testing_data[i,t]=-1
                elif x_til[t]==0:
                    testing_data[i,t]=testing_data[i,t]

            xy=0
        e=sum(np.absolute(testing_data[i,:]-testing_final[:]))

        testing_final[:]=testing_data[i,:]
        if e==0:
            thr=1            
                    
        for f in menory:
            thr_i=0
            thr_i=sum(np.absolute(testing_data[i,:]-x[f,:]))
            if thr_i==0:
                thr=1
                print('testing data'+str(i+1)+'perturbed by '+tit[f])
for f in menory:
    plt.figure(figsize=(3.5,5))
    for i in range(8):
        for y in range(7):
            if testing_data[f,(7*i)+y]==-1:
                plt.scatter(y+1,8-i,color='red',s=50,marker='s')  
    plt.axis([0,7.5,0,8.5])