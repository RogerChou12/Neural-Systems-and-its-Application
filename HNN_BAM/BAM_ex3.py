import matplotlib.pyplot as plt
import data_x_new as s
import numpy as np
import random
np.set_printoptions(linewidth=400) # Ensures long NumPy arrays print in a single line
np.set_printoptions(threshold=np.inf) # Prevents NumPy from truncating arrays

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
time=1000 # Repeats experiments 1000 times for reliable accuracy results
corr=np.zeros([10,7]) # 7 memory sizes(2–8 stored characters), 10 noise levels
for noise in range(1,11): # 1~10 noise points
    for test in range(2,9): # Number of patterns stored in memory: 2 to 8
        corr_t=np.zeros(time) # Stores accuracy for each trial
        for test_time in range(time):        
            m=28 # Split feature vector: first 28 for input, last 28 for output
            w=np.zeros([56-m,m])  # Initialize weight matrix
            memory=random.sample(range(0,8),test)  # choose 2~8 numbers between 0 to 7
            for t in memory:         
                transpose=x[t,range(m,56)].reshape(x[t,range(m,56)].shape[0],1) # Convert row to column (28*1)
                w+=np.multiply(x[t,range(0,m)],transpose) # Compute weights
            testing_data=np.zeros([8,56])
            testing_data[:,:]=x[:,:]

            for i in range(8):
                flipped=random.sample(range(0,56), noise)
                for t in range(noise):
                    testing_data[i,flipped[t]]=-1*testing_data[i,flipped[t]] # Flip the selected bits
            corr_tt=0 # Count correct recall
            for i in memory:
                thr=0 # Convergence flag
                count=0
                testing_final=np.zeros(56) # Store the final stabilized state
                xy=0 # Tracks alternating forward and backward propagation
                while thr==0 and count!=20:
                    count+=1
                    x_til=np.zeros(m)
                    y_til=np.zeros(56-m)
            
                    if xy==0:
                        # Forward propagation: Uses input to predict output
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
                        thr=1   # Stop if no changes         
                                
                for f in memory:
                    thr_i=0
                    thr_i=sum(np.absolute(testing_data[i,:]-x[f,:])) # Compare with original
                    if thr_i==0:
                          corr_tt+=1 # Correct recall count
            corr_t[test_time]=corr_tt/test # success/testing
        corr[noise-1,test-2]=round((sum(corr_t[:])/time)*100,2) # 正確率
print(corr)
index = np.array(range(2,9))
for i in range(10):
    plt.figure()
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
    plt.rcParams['axes.unicode_minus'] = False
    plt.ylabel("percentage")          
    plt.xlabel("menory")            
    plt.title("錯誤點="+str(i+1))
    plt.bar(index, corr[i,:], alpha=0.9, width = 0.6, facecolor = 'skyblue', lw=1)
    for f in range(len(index)):
        plt.annotate(corr[i,f], (-0.2 + index[f], corr[i,f] + 1))