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
time=1000 # Number of times to repeat the experiment for statistical accuracy
corr=np.zeros([10,7]) # 7 memory sizes, 10 mistakes
for noise in range(1,11): # Iterate through noise levels (1 to 10 flipped bits)
    for test in range(2,9): # Number of training samples (2 to 8 characters)
        accuracy_t=np.zeros(time) # Stores accuracy for each trial
        for test_time in range(time):
            memory=random.sample(range(0,8),test) # choose 2~8 numbers between 0 to 7
            w=np.zeros([56,56]) # Initialize 56×56 weight matrix
            for t in memory:         
                g=x[t,:].reshape(x[t,:].shape[0],1) # Convert row to column (56*1)
                w+=np.multiply(x[t,:],g) # compute weights
            for i in range(56):
                w[i,i]=0 # Prevents neurons from self-reinforcing
            testing_data=np.zeros([8,56])
            testing_data[:,:]=x[:,:]

            for i in range(8):
                flipped=random.sample(range(0,56), noise) # 1~10 mistakes
                for t in range(noise):
                    testing_data[i,flipped[t]]=-1*testing_data[i,flipped[t]] # Flip the selected bits
            corr_tt=0 # Counter for correctly recalled patterns
            for i in memory:
                thr=0 # Convergence flag
                count=0
                testing_final=np.zeros(56) # Store the final stabilized state
                while thr==0 and count!=20:
                    count+=1
                    x_til=np.zeros(56)
                    x_til[:]=w[:,:].dot(testing_data[i,:]) # x' = W x testing_data
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
                        corr_tt+=1 # Correct recall count
            accuracy_t[test_time]=corr_tt/test  # 清洗成功數量/測試資料數量
        corr[noise-1,test-2]=round((sum(accuracy_t[:])/time)*100,2) # 正確率

index = np.array(range(2,9))
for i in range(10):
    plt.figure()
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
    plt.rcParams['axes.unicode_minus'] = False
    plt.ylabel("正確率(%)")          
    plt.xlabel("memory") # Number of stored characters           
    plt.title("錯誤點="+str(i+1))
    plt.bar(index, corr[i,:], alpha=0.9, width = 0.6, facecolor = 'skyblue', lw=1)
    for f in range(len(index)):
        plt.annotate(corr[i,f], (-0.2 + index[f], corr[i,f] + 1))

