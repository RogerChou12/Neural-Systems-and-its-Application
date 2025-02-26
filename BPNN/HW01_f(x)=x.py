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
data_x = np.zeros(400)   # Training data X-coordinates
data_y = np.zeros(400)   # Training data Y-coordinates
data_f = np.zeros(400)   # Training data function values

data_x1 = np.zeros(700)  # All data X-coordinates (train + validation + test)
data_y1 = np.zeros(700)  # All data Y-coordinates (train + validation + test)
data_f1 = np.zeros(700)  # All data function values

# Generate 700 random data points within the range [-π, π] for training and testing
for i in range(700):
    data_x1[i] = random.uniform(-1 * math.pi, math.pi)  # Random X value
    data_y1[i] = random.uniform(-1 * math.pi, math.pi)  # Random Y value
    # Function to approximate:
    data_f1[i] = (3 * math.sqrt(data_x1[i] + math.pi) * math.sin(data_x1[i])) + (math.cos(data_y1[i]) / (math.pow(data_y1[i], 2) + 1))

# -----------------------------------------------
# 2. Visualize the generated data in 3D
# -----------------------------------------------

fig = plt.figure()
plt.title('Data')  # Plot title
ax = fig.add_subplot(111, projection='3d')  # Add a 3D subplot

# Scatter plot for training data (Red)
ax.scatter(data_x1[100:400], data_y1[100:400], data_f1[100:400], 
           color='red', s=8, label='train')

# Scatter plot for validation data (Blue)
ax.scatter(data_x1[400:600], data_y1[400:600], data_f1[400:600], 
           color='blue', s=8, label='validation')

# Scatter plot for test data (Green)
ax.scatter(data_x1[600:700], data_y1[600:700], data_f1[600:700], 
           color='green', s=8, label='test')

ax.legend()  # Show legend

# -----------------------------------------------
# 3. Sort training data in ascending order for faster convergence
# -----------------------------------------------

for i in range(400):
    s = 10  # Set a large value to find the minimum
    ti = 0  # Store the index of the minimum element
    for t in range(400):  # Find the smallest X value
        if s > data_x1[t]:
            s = data_x1[t]
            ti = t
    # Store sorted data
    data_x[i] = data_x1[ti]
    data_y[i] = data_y1[ti] # Store corresponding y-values
    data_f[i] = data_f1[ti] # Store corresponding function values
    data_x1[ti] = 10  # Mark as visited
print(data_x)

# -----------------------------------------------
# 4. Initialize Neural Network parameters
# -----------------------------------------------

h = 5  # Number of neurons in the hidden layer
x = 3  # Weight initialization range

# Initialize weights and biases for the hidden and output layers
Wxh = np.zeros(h)  # Weights from input X to hidden layer
Wyh = np.zeros(h)  # Weights from input Y to hidden layer
Whf = np.zeros(h)  # Weights from hidden layer to output

dWxh = np.zeros(h)  # Gradient for Wxh
dWyh = np.zeros(h)  # Gradient for Wyh
dWhf = np.zeros(h)  # Gradient for Whf

# Initialize biases
bf = random.uniform(0, x)  # Bias for output layer
bh = np.zeros(h)  # Bias array for hidden layer

# Randomly initialize weights and biases
for t in range(h):
    Wxh[t] = random.uniform(0, x)
    Wyh[t] = random.uniform(0, x)
    Whf[t]  = random.uniform(0, x)
    bh[t] = random.uniform(0, x)

# -----------------------------------------------
# 5. Train the Neural Network using Backpropagation
# -----------------------------------------------

LRi = 0.08  # Initial learning rate
mse_t = np.zeros(2000)  # Store Mean Squared Error
epoch = 0  # Epoch counter
bf = 0  # Bias for output layer
mse = 10  # Initial MSE

while epoch < 2000:
    e = np.zeros(400)  # Error for each sample
    e_total = 0  # Sum of squared errors
    LR = LRi * math.exp(-1 * (epoch / 100))  # Learning rate decay

    for i in range(400):  # Iterate over training data
        yh = np.zeros(h)  # Hidden layer outputs
        y2 = 0  # Output neuron value

        # Forward propagation: Compute hidden layer activations
        for s in range(h):
            yh[s] = (data_x[i] * Wxh[s] + data_y[i] * Wyh[s] + bh[s])
            yh[s] = max(0, min(1, yh[s]))  # Activation function (ReLU-like)(0~1)
            y2 += yh[s] * Whf[s]

        y2 += bf  # Add bias

        e[i] = ((data_f[i] + 5) / 12) - y2  # Compute error
        delta_f = e[i]  # Output error gradient (data_f-y)*f'(x), f(x)=x

        # Backpropagation
        bf += (delta_f * LR)  # Update output bias

        # Compute gradients and update weights
        delta_h = np.zeros(h)
        for t in range(h):
            dWhf[t] = (delta_f * yh[t]) * LR
            delta_h[t] = delta_f * Whf[t]
            Whf[t] += dWhf[t]

        for f in range(h):
            dWxh[f] = delta_h[f] * data_x[i] * LR
            dWyh[f] = delta_h[f] * data_y[i] * LR
            bh[f] += delta_h[f] * LR
            Wxh[f] += dWxh[f]
            Wyh[f] += dWyh[f]

        e_total += e[i] ** 2  # Accumulate squared error

    mse = math.sqrt(e_total) / 400  # Compute mean squared error
    mse_t[epoch] = mse  # Store MSE
    print(mse)
    epoch += 1

# Plot the Mean Squared Error over epochs
plt.figure()
plt.title('MSE')
plt.plot(mse_t[:])

# -----------------------------------------------
# 6. Testing and Visualization
# -----------------------------------------------

# Compare network predictions with test data
a = np.zeros(700)
for i in range(600, 700): 
    yh1 = np.zeros(h)
    for s in range(h): 
        yh1[s] = (data_x1[i] * Wxh[s] + data_y1[i] * Wyh[s] + bh[s])
        yh1[s] = max(0, min(1, yh1[s]))  # Activation
        a[i] += yh1[s] * Whf[s]
    a[i] += bf
    a[i] = (a[i] * 12) - 5

# Plot test results
plt.figure()
plt.title('TEST epoch=2000')
ax = plt.subplot(111, projection='3d')
ax.scatter(data_x1[600:700], data_y1[600:700], a[600:700], color='black', marker='o')
ax.scatter(data_x1[range(600,700)],data_y1[range(600,700)],data_f1[range(600,700)], color='green',s=15,marker='x')
plt.show()