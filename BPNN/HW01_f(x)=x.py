# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 21:00:54 2021

@author: ray29
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import math
data_x=np.zeros(400)
data_y=np.zeros(400)
data_f=np.zeros(400)
data_x1=np.zeros(700)
data_y1=np.zeros(700)
data_f1=np.zeros(700)
for i in range(700):
    data_x1[i]=random.uniform(-1*(math.pi),math.pi)
    data_y1[i]=random.uniform(-1*(math.pi),math.pi)
    data_f1[i]=(3*math.sqrt(data_x1[i]+math.pi)*math.sin(data_x1[i]))+\
        (math.cos(data_y1[i])/(math.pow(data_y1[i],2)+1))

fig = plt.figure()
plt.title('Data')
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data_x1[range(100,400)],data_y1[range(100,400)],data_f1[range(100,400)], color='red',s=8,label='train')
ax.scatter(data_x1[range(400,600)],data_y1[range(400,600)],data_f1[range(400,600)], color='blue',s=8,label='validation')
ax.scatter(data_x1[range(600,700)],data_y1[range(600,700)],data_f1[range(600,700)], color='green',s=8,label='test')
ax.legend()
for i in range(400):
    s=10
    ti=0
    for t in range(400):     # training data由大到小排列
        if s > data_x1[t]:
            s=data_x1[t]
            ti=t
    data_x[i]=data_x1[ti]
    data_y[i]=data_y1[ti]
    data_f[i]=data_f1[ti]
    data_x1[ti]=10
print(data_x)

h=5
x=3
Wxh=np.zeros(h)
Wyh=np.zeros(h)
Whf=np.zeros(h)
init_Wxh=np.zeros(h)
init_Wyh=np.zeros(h)
init_Whf=np.zeros(h)
ep1_Wxh=np.zeros(h)
ep1_Wyh=np.zeros(h)
ep1_Whf=np.zeros(h)
dWxh=np.zeros(h)
dWyh=np.zeros(h)
dWhf=np.zeros(h)
bf=init_bf=random.uniform(0,x)
init_bh=np.zeros(h)
bh=np.zeros(h)
ep1_bh=np.zeros(h)
for t in range(h):
    Wxh[t]=init_Wxh[t]=random.uniform(0,x)
    Wyh[t]=init_Wyh[t]=random.uniform(0,x)
    Whf[t]=init_Whf[t]=random.uniform(0,x)
    
    bh[t]=init_bh[t]=random.uniform(0,x)

LRi=0.08
mse_t=np.zeros(2000)
epoch=0
bf=0
mse=10
while epoch<2000:
    e=np.zeros(400)
    e_total=0
    LR=LRi*math.exp(-1*(epoch/100))
   
    for i in range(400):
        yh=np.zeros(h)

        y2=0
        for s in range(h):
            yh[s]=(data_x[i]*Wxh[s]+data_y[i]*Wyh[s]+bh[s])
           
            if yh[s]>1:
                yh[s]=1
            if yh[s]<0:
                yh[s]=0   
            y2+=yh[s]*Whf[s]
        y2+=bf
       
        e[i]=((data_f[i]+5)/12)-y2
        #print(y2)
        delta_f=e[i]
      
        bf+=(delta_f*LR)
       
        delta_h=np.zeros(h)
        for t in range(h):
            
            dWhf[t]=(delta_f*yh[t])*LR
            
            delta_h[t]=delta_f*Whf[t]
           
            Whf[t]+=dWhf[t]
            if epoch==0:
                ep1_Whf[t]=Whf[t]
                ep1_bf=bf
        for f in range(h):
            
            dWxh[f]=delta_h[f]*data_x[i]*LR
            dWyh[f]=delta_h[f]*data_y[i]*LR
           
           
            
            bh[f]+=delta_h[f]*LR
            Wxh[f]+=dWxh[f]
            Wyh[f]+=dWyh[f]
            if epoch==0:
                ep1_bh[f]=bh[f]
                ep1_Wxh[f]=Wxh[f]
                ep1_Wyh[f]=Wyh[f]
          
        #print(bh)  
       
        e_total+=e[i]*e[i]
    mse=math.sqrt(e_total)/400
    
    mse_t[epoch]=mse
    print(mse)
    epoch+=1
plt.figure()
plt.title('MSE')
plt.plot(mse_t[:])
print(init_Whf)
print(init_Wxh) 
print(init_Wyh)
print(init_bh)
print(init_bf)

print(Whf)
print(Wxh) 
print(Wyh)
print(bh)
print(bf)
ep1=np.zeros(700)
yh1=np.zeros(h)        
for i in range(600,700): 
    for s in range(h): 
        yh1[s]=(data_x1[i]*ep1_Wxh[s]+data_y1[i]*ep1_Wyh[s]+ep1_bh[s])
        if yh1[s]>1:
            yh1[s]=1
        if yh1[s]<0:
            yh1[s]=0   
        ep1[i]+=yh1[s]*ep1_Whf[s]
    ep1[i]+=ep1_bf
    ep1[i]=(ep1[i]*12)-5
fi = plt.figure()
ap = fi.add_subplot(111, projection='3d')
plt.title('TEST epoch=1')
ap.scatter(data_x1[range(600,700)],data_y1[range(600,700)],ep1[range(600,700)], color='black',s=15,marker='o',label='BPNN')
ap.scatter(data_x1[range(600,700)],data_y1[range(600,700)],data_f1[range(600,700)], color='green',s=15,marker='x',label='test')
a=np.zeros(700)
yh1=np.zeros(h)        
for i in range(600,700): 
    for s in range(h): 
        yh1[s]=(data_x1[i]*Wxh[s]+data_y1[i]*Wyh[s]+bh[s])
        if yh1[s]>1:
            yh1[s]=1
        if yh1[s]<0:
            yh1[s]=0   
        a[i]+=yh1[s]*Whf[s]
    a[i]+=bf
    a[i]=(a[i]*12)-5
fi = plt.figure()
plt.title('TEST epoch=2000')
ap = fi.add_subplot(111, projection='3d')
ap.scatter(data_x1[range(600,700)],data_y1[range(600,700)],a[range(600,700)], color='black',s=15,marker='o')
ap.scatter(data_x1[range(600,700)],data_y1[range(600,700)],data_f1[range(600,700)], color='green',s=15,marker='x')
plt.show()