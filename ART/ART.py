# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import data_x as s
import numpy as np

word = ['井','五','亘','交','亢','云','了','予','乙','也','乏','乃','久','人','什','介']
title = ["" for x in range(16)]
data = np.zeros([16,70])

fig = plt.figure()  # width and height of figures
ax = fig.subplots(2,8)
ax = ax.flatten()

for x in range(16):
    data[x,:] = s.data(x)
    for i in range(10):
        for j in range(7):
            if data[x,i*7+j] == 1:
                ax[x].scatter(j+1, 10-i, color='red', s=50, marker='o')
    ax[x].axis([0,7.5,0,10.5])
fig.show()

#initial weights
w_wave = np.zeros([16,70])
w_wave[0,:] = data[0,:]
w = np.zeros([16,70])
epoch = 0
count = 1
thr = 0.6
for epoch in range(10):
    for x in range(16):
        y1 = np.ones(count)
        y = np.zeros(count)
        sort = 0
        wc = 0
        while sort == 0:
            for c in range(count):
                wc = 1/(0.5+sum(w_wave[c,:]))
                w[c,:] = wc*w_wave[c,:]
                y[c] = sum(w[c,:]*data[x,:])*y1[c]
            max_index = np.argmax(y)
            v = sum(w_wave[max_index,:]*data[x,:])/abs(sum(data[x,:]))
            if v > thr:
                w_wave[max_index,:] = w_wave[max_index,:]*data[x,:]
                w[max_index,:] = w_wave[max_index,:]/(0.5+sum(w_wave[max_index,:]))
                if epoch == 9:
                    title[max_index] += word[x]
                sort = 1
            else:
                y1[max_index] = 0
            if sum(y1) == 0:
                count += 1
                w_wave[count-1] = data[x,:]
                if epoch == 9:
                    title[count-1] += word[x]
                sort = 1
    print('epoch='+str(epoch))
print('After converging:')
print(title)

for f in range(count):
    plt.figure(figsize=(3.5,5))  # width and height of figures
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] #設定字型(支援中文)
    plt.rcParams['axes.unicode_minus'] = False  # 正常顯示字元
    plt.title(str(title[f]),size=30)
    for i in range(10):
        for j in range(7):
            if w_wave[f,i*7+j] == 1:
                plt.scatter(j+1, 10-i, color='blue', s=50, marker='o')
    plt.axis([0,7.5,0,10.5])
plt.show()