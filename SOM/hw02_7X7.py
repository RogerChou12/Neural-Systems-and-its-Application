# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import random
import math
t = 49
#R=3

x = np.array([[1,0,0,0,0],[2,0,0,0,0],[3,0,0,0,0],[4,0,0,0,0],[5,0,0,0,0],
             [3,1,0,0,0],[3,2,0,0,0],[3,3,0,0,0],[3,4,0,0,0],[3,5,0,0,0],
             [3,3,1,0,0],[3,3,2,0,0],[3,3,3,0,0],[3,3,4,0,0],[3,3,5,0,0],[3,3,6,0,0],[3,3,7,0,0],[3,3,8,0,0],
             [3,3,3,1,0],[3,3,3,2,0],[3,3,3,3,0],[3,3,3,4,0],
             [3,3,6,1,0],[3,3,6,2,0],[3,3,6,3,0],[3,3,6,4,0],
             [3,3,6,2,1],[3,3,6,2,2],[3,3,6,2,3],[3,3,6,2,4],[3,3,6,2,5],[3,3,6,2,6]])
tit = np.array(['A','B','C','D','E',
             'F','G','H','I','J',
             'K','L','M','N','O','P','Q','R',
             'S','T','U','V',
             'W','X','Y','Z',
             '1','2','3','4','5','6'])
topo_map = np.array([[1,1],[1,2],[1,3],[1,4],[1,5],[1,6],[1,7],
               [2,1],[2,2],[2,3],[2,4],[2,5],[2,6],[2,7],
               [3,1],[3,2],[3,3],[3,4],[3,5],[3,6],[3,7],
               [4,1],[4,2],[4,3],[4,4],[4,5],[4,6],[4,7],
               [5,1],[5,2],[5,3],[5,4],[5,5],[5,6],[5,7],
               [6,1],[6,2],[6,3],[6,4],[6,5],[6,6],[6,7],
               [7,1],[7,2],[7,3],[7,4],[7,5],[7,6],[7,7]])
topo = np.zeros([t,5])
for i in range(t):
    topo[i,0] = random.uniform(0,4)
    topo[i,1] = random.uniform(0,4)
    topo[i,2] = random.uniform(0,4)
    topo[i,3] = random.uniform(0,4)
    topo[i,4] = random.uniform(0,4)
print ('beginning weight')
print (topo)
d = np.zeros(t)
lr_t = 0
for tim in range(3):
    R0 = 3-tim
    epoch = 0
    
    while epoch <75:
        LR = 0.1*math.exp(-1*(lr_t/1000))
        if R0 == 1:
            R = R0      # radius
        else:
            t1 = 1000/math.log(R0)
            R = R0*math.exp(-(lr_t/t1))      # radius
        for k in range(32):
            for e in range(t):
                d[e] = 0
                for s in range(5):
                    d[e] += np.power((x[k,s]-topo[e,s]),2)     
                d[e] = math.sqrt(d[e])
            min_index = np.argmin(d)
            min_value = min(d)
            ##print(min_index)
            ##print(topo_map[min_index,:])
            min_map_i = topo_map[min_index,0]
            min_map_j = topo_map[min_index,1]
            #rint(len(topo_map))
            h_qj = np.zeros(t)
            for i in range(t):
                h_qj[i] = math.exp(-1*((np.power(topo_map[i,0] - min_map_i,2) + np.power(topo_map[i,1] - min_map_j,2))/(2*R*R)))
                for j in range(5):
                    topo[i,j] += LR*h_qj[i]*(x[k,j]-topo[i,j])
        epoch += 1
        lr_t += 1
    print('epoch=' + str(lr_t) + '時weight:')
    print(topo)
    #繪圖
    drawn_tit = ['' for q in range(t)]
    dr_x = np.zeros([32,2])
    for i in range(len(x)):
        x_tra_min = np.zeros(t)
        for j in range(t):
            for k in range(5):
                x_tra_min[j] += np.power((x[i,k]-topo[j,k]),2)
            x_tra_min[j] = math.sqrt(x_tra_min[j])
        x_min_index = np.argmin(x_tra_min)
        drawn_tit[x_min_index] += ' '+tit[i]
            
        
    plt.figure()
    plt.title('R='+str(R)+',epoch='+str(lr_t)+',LR='+str(round(LR, 3)))
    plt.scatter(topo_map[:,0],topo_map[:,1], color='red',s=5)
    for i in range(t):
        plt.annotate(drawn_tit[i],(topo_map[i,0],topo_map[i,1]),fontsize=15)
    #plt.legend()
plt.show()