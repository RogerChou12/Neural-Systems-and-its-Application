# Neural System and its Application
## BPNN
1. Simulate funtion f(x,y) with BPNN
   >$`f(x,y)=3\sqrt{x+\pi}\sin{x}+\frac{\cos{y}}{y^2+1}, (x,y)\in(-\pi,\pi)`$  
2. 5 neurons in hidden layer  
3. Learning Rate decays
   >$`\eta=\eta_0*\exp(-\frac{epoch}{1000})`$  
4. Different activation function
   >$`f(x)=x`$ (ReLU-like), $`f(x)=\tanh{x}`$  
5. Compute error between between outputs and targets, then adjest bias and weights of output layer and hidden layer
   >$`\Delta b_{output}=(f(x,y)-y_{output})f'(x)`$ => $`b_{output} = b_{output}+\Delta b_{output}*\eta`$  
   >$`\Delta W_{output}=\Delta b_{output}*y_{hidden}*\eta`$ => $`W_{output}=W_{output}+\Delta W_{output}`$  
   >$`\Delta b_{hidden}=\Delta b_{output}*W_{output}*f'(x)`$ => $`b_{hidden}=b_{hidden}+\Delta b_{hidden}*\eta`$  
   >$`\Delta W_{hidden-x}=\Delta b_{hidden}*x*\eta`$ => $`W_{hidden-x}=W_{hidden-x}+\Delta W_{hidden-x}`$  
   >$`\Delta W_{hidden-y}=\Delta b_{hidden}*y*\eta`$ => $`W_{hidden-y}=W_{hidden-y}+\Delta W_{hidden-y}`$  
6. Compute Mean Squared Error (MSE)
   >$`MSE=\sqrt{\sum (f(x,y)-y_{output})^2}/numbers_{training-data}`$  

## Self-Organizing Map (SOM)
1. Training data (X): 32, each with 5 features (A to Z & 1 to 6)
2. Numbers of neurons(M): 7x7 or 10x7
3. Training the SOM for 3 Iterations and reducing the neighborhood radius each time
4. Train for 75 Epochs Per Round  
   4-1. Learning Rate decays
      >$`\eta=\eta_0*\exp(-1\frac{epoch}{1000})`$
      
   4-2. The neighborhood radius is reduced  
      >$`R=R_0*\exp(-\frac{epoch}{\tau_1}), \tau_1=\frac{1000}{\log(R_0)}`$
      
   4-3. Find the winner neuron with the smallest Euclidean distance between inputs and neurons  
      > $`q(X)=\min_{\forall j}(\|(X-W_j)\|_2)`$, j=1,2,...,M  
      > $`h_{qj}(epoch)=\exp(-\frac{\|(r_j-r_q)\|^2}{2R^2})`$, $`r_q`$=position of winner neuron in topology, $`r_j`$=neighborhood neuron
   
   4-4. Update weights  
      >$`W_j(epoch+1)=W_j(epoch)+\eta*h_{qj}(epoch)*(X(epoch)-W_j(epoch))`$  

## Learning Vector Quantization (LVQ)
1. Training data: 32, each with 5 features
2. 6 classes (M) = 6 neurons
3. Learning Rate decays  
   >$`\eta=\eta_0*\exp(-1\frac{epoch}{1000})`$  
5. Compute Euclidean distance between inputs and neurons  
   >$`q(X)=\min_{\forall j}(\|(X-W_j)\|_2)`$, j=1,2,...,M  
7. Find the winner neuron. If the class matches, strengthen weights.  
   >$`C_{w_q} = C_{x_j}, W_q(epoch+1)=W_q(epoch)+\eta*(X-W_q(epoch))`$  
   >$`C_{w_q} \neq C_{x_j}, W_q(epoch+1)=W_q(epoch)-\eta*(X-W_q(epoch))`$  
## Adaptive Resonance Theory (ART)
1. Training data:16 Chinese characters, each with 70 binary features  
2. Initialize the weights of the first cluster with the first character  
   >$`W_0=X_0`$  
3. Find he most similar cluster  
   >$`Y[cluster]=(\displaystyle\sum_{i=0}^{69} W[cluster,i]*X[cluster,i])*W_{norm}[cluster]*y_{active}[cluster]`$  
   >$`W_{norm}=\frac{1}{\sum W_{cluster}+0.5}`$  
   >$`index_{winner}=argmax(Y)`$  
4. Compute similarity score  
   >$`v=\frac{\sum_{i=0}^{69} W[index_{winner},i]*X[character,i]}{\sum_{i=0}^{69} X[character,i]}`$  
