import matplotlib.pyplot as plt
import data_x_new as s
import numpy as np
import random
np.set_printoptions(linewidth=400)
np.set_printoptions(threshold=np.inf)

x=np.zeros([8,56])
tit=['五','亢','云','了','乙','乃','久','人']
for i in range(8):
    x[i,:]=s.data(i)
    for c in range(56):
        if x[i,c]==1:
            x[i,c]=-1
        elif x[i,c]==0:
            x[i,c]=1
time=1000
corr=np.zeros([10,7])
for ram in range(1,11):
    for test in range(2,9):
        corr_t=np.zeros(time)
        for test_time in range(time):
            menory=random.sample(range(0,8),test)
            w=np.zeros([56,56])
            for t in menory:         
                g=x[t,:].reshape(x[t,:].shape[0],1)
                w+=np.multiply(x[t,:],g)
            for i in range(56):
                w[i,i]=0
            testing_data=np.zeros([8,56])
            testing_data[:,:]=x[:,:]

            for i in range(8):
                a=random.sample(range(0,56), ram)
                for t in range(ram):
                    testing_data[i,a[t]]=-1*testing_data[i,a[t]] 
            corr_tt=0
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
                        corr_tt+=1
            corr_t[test_time]=corr_tt/test # success/testing
        corr[ram-1,test-2]=round((sum(corr_t[:])/time)*100,2) #正確率
    #print(corr)
    

#color=['lawngreen','limegreen','lime','mediumspringgreen','mediumturquoise','aqua','deepskyblue','skyblue','dodgerblue','cornflowerblue']
index = np.array(range(2,9))
for i in range(10):
    plt.figure()
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
    plt.rcParams['axes.unicode_minus'] = False
    plt.ylabel("正確率(%)")          
    plt.xlabel("menory")            
    plt.title("錯誤點="+str(i+1))
    plt.bar(index, corr[i,:], alpha=0.9, width = 0.6, facecolor = 'skyblue', lw=1)
    for f in range(len(index)):
        plt.annotate(corr[i,f], (-0.2 + index[f], corr[i,f] + 1))

