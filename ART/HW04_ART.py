
import matplotlib.pyplot as plt
import data_x as s
import numpy as np
drawn_tit=['' for q in range(16)]
tit=np.array(['井','五','亘','交','亢','云','了','予','乙','也','乏','乃','久','人','什','介'])

x=np.zeros([16,70])

for i in range(16):
    x[i,:]=s.data(i)
for f in range(16):
    plt.figure(figsize=(3.5,5))  # width and height of figures
    for i in range(10):
       for y in range(7):
           if x[f,(7*i)+y]==1:   # If the value is 1, plot a square
            # In image representation, the top-left corner is (0,0) in the matrix.
            # But the origin(0,0) of Matplotlib's default coordinate system is at bottom-left.
             plt.scatter(y+1,10-i,color='red',s=50,marker='s')  
    plt.axis([0,7.5,0,10.5])  #x-axis(min,max) y-axis(min,max)

w=np.zeros([20,70])
w[0,:]=x[0,:]
epoch=0
thr=0.8   # thr=0.4 0.6 0.8 # Similarity threshold (higher = stricter clustering)
w_count=1 # Number of clusters
epoch_t=8 # Training epochs
while epoch < epoch_t:    
    for t in range(16):
        w_tid=np.zeros(w_count)
        for f in range(w_count):
            w_tid[f]=1/(sum(w[f,:])+0.5)
        y=np.zeros(w_count)
        y_1=np.ones(w_count)
        test=0
        count=0
        while test ==0:
            for time in range(w_count):
                y[time]=sum(w[time,:]*x[t,:])*w_tid[time]*y_1[time]
            
            max_index=np.argmax(y) # The most similar cluster
            v=sum(w[max_index,:]*x[t,:])/sum(x[t,:])   # Similarity score

            if v >= thr: # Merges the character into the cluster if similarity is high enough
                w[max_index,:]=w[max_index,:]*x[t,:]
                test=1
                if epoch ==(epoch_t-1): # If it's the final epoch, the character is added to 'drawn_tit'
                    drawn_tit[max_index]+=tit[t]
            else:
                y_1[max_index]=0
            if sum(y)==0: # If no suitable cluster is found, a new cluster is created
                w[w_count,:]=x[t,:]
                if epoch ==(epoch_t-1): 
                    drawn_tit[w_count]+=tit[t]
                w_count+=1
                test=1
    epoch+=1
    print(epoch)
print(drawn_tit)    
for f in range(w_count):
    plt.figure(figsize=(3.5,5))
    #figure tilte顯示中文字
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] # Set font for Chinese text
    plt.rcParams['axes.unicode_minus'] = False  # Ensure proper character display
    plt.title(str(drawn_tit[f]),size=30)
    for i in range(10):
       for y in range(7):
           if w[f,(7*i)+y]==1:
             plt.scatter(y+1,10-i,color='green',s=50,marker='s')  # Plot characters
    plt.axis([0,7.5,0,10.5])   
plt.show()
          
            
        
