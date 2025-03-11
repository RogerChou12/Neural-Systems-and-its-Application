
import matplotlib.pyplot as plt
import data_x as s
import numpy as np
drawn_tit=['' for q in range(16)]
tit=np.array(['井','五','亘','交','亢','云','了','予','乙','也','乏','乃','久','人','什','介'])

x=np.zeros([16,70]) # 16 characters, each with a 70-dimensional binary feature vector

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

w=np.zeros([20,70]) # Stores learned cluster centers (max 20 clusters)
w[0,:]=x[0,:] # Initialize the first cluster with the first character's pattern
epoch=0
thr=0.8   # thr=0.4 0.6 0.8 # Similarity threshold (higher = stricter clustering)
clusters=1 # Number of clusters
epoch_total=8 # Training epochs
while epoch < epoch_total:    
    for character in range(16):
        w_norm=np.zeros(clusters)
        for f in range(clusters):
            w_norm[f]=1/(sum(w[f,:])+0.5)
        y=np.zeros(clusters) # Cluster activation values
        y_1=np.ones(clusters) # Activation status for clusters
        assigned=0 # determine if a character is assigned to a cluster
        count=0
        while assigned ==0:
            for cluster in range(clusters):
                y[cluster]=sum(w[cluster,:]*x[character,:])*w_norm[cluster]*y_1[cluster]
            
            max_index=np.argmax(y) # The most similar cluster
            v=sum(w[max_index,:]*x[character,:])/sum(x[character,:])   # Similarity score

            if v >= thr: # Merges the character into the cluster if similarity is high enough
                w[max_index,:]=w[max_index,:]*x[character,:] # Update weights
                assigned=1
                if epoch ==(epoch_total-1): # If it's the final epoch, the character is added to 'drawn_tit'
                    drawn_tit[max_index]+=tit[character]
            else: # Similarity score is too low
                y_1[max_index]=0 # Temporarily deactivate the cluster
            
            if sum(y)==0: # If no suitable cluster is found, a new cluster is created
                w[clusters,:]=x[character,:]
                if epoch ==(epoch_total-1): 
                    drawn_tit[clusters]+=tit[character]
                clusters+=1
                assigned=1
    epoch+=1
    print(epoch)
print(drawn_tit)    
for f in range(clusters):
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
          
            
        
