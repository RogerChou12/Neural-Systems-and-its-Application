# Neural System and its Application
## BPNN
1. Generate 700 random data points within the range [-π, π] for training and testing. (Train: 400, Validation: 200, Test: 100)
2. Sort training data in ascending order for faster convergence and reducing weight oscillations
3. Train the Neural Network using Backpropagation
   3-1. 5 neurons in hidden layer  
   3-2. Learning Rate decays exponentially  
   3-3. different activation function: f(x)=x (ReLU-like), f(x)=tanh(x)  
   3-4. mean squared error  

## SOM
1. Training data: 32, each with 5 features (A to Z & 1 to 6)
2. Number of neurons: 7x7 or 10x7
3. Training the SOM for 3 Iterations and reducing the neighborhood radius each time
4. Train for 75 Epochs Per Round
   4-1. Learning Rate decays exponentially over time  
   4-2. The neighborhood radius is reduced gradually to focus on fine-tuning  
   4-3. Compute Euclidean distance between inputs and neurons  
   4-4. Find the position of the neuron with the smallest distance  
   4-5. Update weights with LR * h_qj[i] * (x[k] - topo[i])  

## LVQ
1. Training data: 32, each with 5 features
2. 6 classes = 6 neurons
3. Compute Euclidean distance between inputs and neurons
4. Find the winner neuron. If class matches, strengthen weights.
