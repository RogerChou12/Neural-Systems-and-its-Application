# Neural System and its Application
## BPNN
1. Simulate funtion $`f(x,y)=3\sqrt{x+\pi}\sin{x}+\frac{\cos{y}}{y^2+1}, (x,y)\in(-\pi,\pi)`$ with BPNN
2. 5 neurons in hidden layer  
3. Learning Rate decays $`\eta=\eta_0*\exp(-1\frac{epoch}{1000})`$  
4. Different activation function: $`f(x)=x`$ (ReLU-like), $`f(x)=\tanh{x}`$
5. Compute error between between outputs and targets, then adjest bias and weights of output layer and hidden layer  
6. Compute Mean Squared Error (MSE)

## Self-Organizing Map (SOM)
1. Training data: 32, each with 5 features (A to Z & 1 to 6)
2. Numbers of neurons(M): 7x7 or 10x7
3. Training the SOM for 3 Iterations and reducing the neighborhood radius each time
4. Train for 75 Epochs Per Round  
   4-1. Learning Rate decays $`\eta=\eta_0*\exp(-1\frac{epoch}{1000})`$  
   4-2. The neighborhood radius is reduced $`R=R_0*\exp(-\frac{epoch}{\tau_1}), \tau_1=\frac{1000}{\log(R_0)}`$  
   4-3. Find the winner neuron with the smallest Euclidean distance between inputs and neurons  
      > $`q(X)=\min_{\forall j}(\|(X-W_j)\|_2)`$, j=1,2,...,M  
      > $`h_{qj}(epoch)=\exp(-\frac{\|(r_j-r_q)\|^2}{2R^2})`$, $`r_q`$=position of winner neuron in topology, $`r_j`$=neighborhood neuron
   
   4-4. Update weights with $`W_j(epoch+1)=W_j(epoch)+\eta*h_{qj}(epoch)*(X(epoch)-W_j(epoch)`$  

## Learning Vector Quantization (LVQ)
1. Training data: 32, each with 5 features
2. 6 classes = 6 neurons
3. Compute Euclidean distance between inputs and neurons
4. Find the winner neuron. If the class matches, strengthen weights.
