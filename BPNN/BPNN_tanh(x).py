# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import math

# -----------------------------------------------
# 1. Generate synthetic data for function approximation
# -----------------------------------------------

# Initialize arrays to store data
data_x = np.zeros(700)
data_y = np.zeros(700)
data_f = np.zeros(700)
def tanh(x):
    t = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    return t
for i in range(700):
    data_x[i] = random.uniform(-1*(math.pi), math.pi)
    data_y[i] = random.uniform(-1*(math.pi), math.pi)
    data_f[i] = (3*math.sqrt(data_x[i] + math.pi)*math.sin(data_x[i])) + (math.cos(data_y[i])/(math.pow(data_y[i],2) + 1))

# -----------------------------------------------
# 2. Visualize the generated data in 3D
# -----------------------------------------------

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data_x[range(100,400)],data_y[range(100,400)],data_f[range(100,400)], color='red',s=8)
ax.scatter(data_x[range(400,600)],data_y[range(400,600)],data_f[range(400,600)], color='blue',s=8)
ax.scatter(data_x[range(600,700)],data_y[range(600,700)],data_f[range(600,700)], color='green',s=8)

# -----------------------------------------------
# 3. Initialize Neural Network parameters
# -----------------------------------------------

h = 5 # Number of neurons in the hidden layer
x = 3
# Initialize weights and biases for the hidden and output layers
Wxh = np.zeros(h)
Wyh = np.zeros(h)
Whf = np.zeros(h)

dWxh = np.zeros(h)
dWyh = np.zeros(h)
dWhf = np.zeros(h)

bf = random.uniform(0,x)
bh = np.zeros(h)
for t in range(h):
    Wxh[t] = random.uniform(0,x) #weights of input x to hidden layer
    Wyh[t] = random.uniform(0,x) #weights of input  to hidden layer
    Whf[t] = random.uniform(0,x) #weights of hidden layer to output
    bh[t] = random.uniform(0,x) #bias of hidden layer

# -----------------------------------------------
# 4. Train the Neural Network using Backpropagation
# -----------------------------------------------

LRi = 0.3 # Initial learning rate
mse_t = np.zeros(2000) # Store Mean Squared Error
epoch = 0
while epoch < 2000:
    e = np.zeros(400) # Error for each sample
    e_total = 0 # Sum of squared errors
    LR = LRi*math.exp(-1*(epoch/1000))
    for i in range(400):
        yh = np.zeros(h) # Hidden layer outputs
        y2 = 0 # Output neuron value

        # Forward propagation: Compute hidden layer
        for s in range(h):
            yh[s] = tanh(data_x[i]*Wxh[s] + data_y[i]*Wyh[s] + bh[s])
            y2 += yh[s]*Whf[s]
        y2 = tanh(y2 + bf)
        
        e[i] = ((data_f[i] + 5)/12) - y2 # Compute error
        delta_f = e[i]*(1 - y2*y2)  # (data_f-y)*f'(x), f(x)=tanh(x)
        
        # Backpropagation
        bf += (delta_f*LR) # Update output bias
        
        # Compute gradients and update weights
        delta_h = np.zeros(h)
        for t in range(h):
            dWhf[t] = (delta_f*yh[t])*LR  # (data_f-y)*f'(x) * output of neurons * LR
            delta_h[t] = delta_f*Whf[t]*(1-yh[t]*yh[t])
            Whf[t] += dWhf[t]
        
        for f in range(h):
            dWxh[f] = delta_h[f]*data_x[i]*LR
            dWyh[f] = delta_h[f]*data_y[i]*LR
            bh[f] += delta_h[f]*LR
            Wxh[f] += dWxh[f]
            Wyh[f] += dWyh[f]
        
        e_total += e[i] ** 2 # Accumulate squared error

    mse=math.sqrt(e_total)/400
    mse_t[epoch]=mse
    print(mse)
    epoch+=1

# Plot the Mean Squared Error over epochs
plt.figure()
plt.title("MSE")
plt.plot(mse_t[:])
print(Whf)
print(Wxh) 
print(Wyh)
print(bh)
print(bf)

# -----------------------------------------------
# 5. Testing and Visualization
# -----------------------------------------------

# Compare network predictions with test data
a = np.zeros(700)        
for i in range(600,700):
    yh1 = np.zeros(h) 
    for s in range(h): 
        yh1[s] = tanh(data_x[i]*Wxh[s]+data_y[i]*Wyh[s]+bh[s]) 
        a[i] += yh1[s]*Whf[s]
    a[i] = tanh(a[i]+bf)
    a[i] = (a[i]*12)-5

# Plot test results
fi = plt.figure()
ap = fi.add_subplot(111, projection='3d')
plt.title('TEST epoch=2000')
ap.scatter(data_x[range(600,700)],data_y[range(600,700)],a[range(600,700)],s=15,marker='o',facecolor=(0,0,0,0), edgecolor='black')
ap.scatter(data_x[range(600,700)],data_y[range(600,700)],data_f[range(600,700)], color='green',s=15,marker='x')
plt.show()