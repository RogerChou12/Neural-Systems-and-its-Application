import matplotlib.pyplot as plt
import data_x_new as s
import numpy as np
import random
np.set_printoptions(linewidth=400) # Ensures long NumPy arrays print in a single line
np.set_printoptions(threshold=np.inf) # Prevents NumPy from truncating arrays
memory=random.sample(range(0,8),3) # choose 3 data randomly

x=np.zeros([8,56])
tit=['五','亢','云','了','乙','乃','久','人'] # Chinese character labels
for i in range(8):
    x[i,:]=s.data(i)
    # Converts 1s to -1 and 0s to 1
    for c in range(56):
        if x[i,c]==1:
            x[i,c]=-1
        elif x[i,c]==0:
            x[i,c]=1

m=28
w=np.zeros([56-m,m])
for t in memory:
    transpose=x[t,range(m,56)].reshape(x[t,range(m,56)].shape[0],1)
    w+=np.multiply(x[t,range(0,m)],transpose) # Compute weights

testing_data=np.zeros([8,56])
testing_data[:,:]=x[:,:]
noise=6  # 6 noise points
for i in range(8):
    flipped=random.sample(range(0,56), noise)
    for t in range(noise):
        testing_data[i,flipped[t]]=-1*testing_data[i,flipped[t]] # Flip the selected bits
for f in memory:
    plt.figure(figsize=(3.5,5))
    for i in range(8):
       for y in range(7):
           if testing_data[f,(7*i)+y]==-1:
             plt.scatter(y+1,8-i,color='red',s=50,marker='s')  
    plt.axis([0,7.5,0,8.5])  

testing_data1=np.zeros([8,56])
testing_data1[:,:]=testing_data[:,:]

for i in memory:  # X(0)->Y(1)->...
    thr=0 # Convergence flag
    count=0
    testing_final=np.zeros(56) # Store the final stabilized state
    xy=0 # Tracks alternating forward and backward propagation
    while thr==0 and count!=20:
        count+=1
        x_til=np.zeros(m)
        y_til=np.zeros(56-m)

        # Forward propagation: Uses input to predict output
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
        # Backward propagation: Uses output to reconstruct input
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
        error=sum(np.absolute(testing_data[i,:]-testing_final[:]))

        testing_final[:]=testing_data[i,:]
        if error==0:
            thr=1  # Stop if no changes
                    
        for f in memory:
            thr_i=0
            thr_i=sum(np.absolute(testing_data[i,:]-x[f,:])) # Compare with original
            if thr_i==0:
                thr=1
                print('testing data '+str(i+1)+' perturbed by '+tit[f])

for i in memory:   # Y(0)->X(1)->...
    thr=0
    count=0
    testing_final=np.zeros(56)
    xy=1
    while thr==0 and count!=20:
        count+=1
        x_til=np.zeros(m)
        y_til=np.zeros(56-m)

        # Forward propagation: Uses input to predict output
        if xy==0:
            y_til[:]=w[:,:].dot(testing_data1[i,range(0,m)])
            for t in range(m,56):
                if y_til[t-m]>0:
                    testing_data1[i,t]=1
                elif y_til[t-m]<0:
                    testing_data1[i,t]=-1
                elif y_til[t-m]==0:
                    testing_data1[i,t]=testing_data1[i,t]

            xy=1
        # Backward propagation: Uses output to reconstruct input
        elif xy!=0:
            x_til[:]=(w[:,:].transpose()).dot(testing_data1[i,range(m,56)])
            for t in range(m):
                if x_til[t]>0:
                    testing_data1[i,t]=1
                elif x_til[t]<0:
                    testing_data1[i,t]=-1
                elif x_til[t]==0:
                    testing_data1[i,t]=testing_data1[i,t]

            xy=0
        error=sum(np.absolute(testing_data1[i,:]-testing_final[:]))

        testing_final[:]=testing_data1[i,:]
        if error==0:
            thr=1 # Stop if no changes
                    
        for f in memory:
            thr_i=0
            thr_i=sum(np.absolute(testing_data1[i,:]-x[f,:])) # Compare with original
            if thr_i==0:
                thr=1
                print('testing data '+str(i+1)+' perturbed by '+tit[f])
for f in memory:
    plt.figure(figsize=(3.5,5))
    for i in range(8):
        for y in range(7):
            if testing_data[f,(7*i)+y]==-1:
                plt.scatter(y+1,8-i,color='red',s=50,marker='s')  
    plt.axis([0,7.5,0,8.5])
for f in memory:
    plt.figure(figsize=(3.5,5))
    for i in range(8):
        for y in range(7):
            if testing_data1[f,(7*i)+y]==-1:
                plt.scatter(y+1,8-i,color='red',s=50,marker='s')  
    plt.axis([0,7.5,0,8.5])