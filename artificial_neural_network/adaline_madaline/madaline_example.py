#Adaline neural network
import numpy as np
import pandas as pd

def activation_fn(z):
    if z>=0:
        return 1  
    else:
        return -1

def Madaline(Input, Target, lr, epoch):
    weight = np.random.random((Input.shape[1],Input.shape[1]))
    bias   = np.random.random(Input.shape[1])
    
    w = np.array([0.5 for i in range(weight.shape[1])])
    b = 0.5
    k = 0
    while k<epoch: 
        error = []
        z_input = np.zeros(bias.shape[0])
        z = np.zeros(bias.shape[0])
        for i in range(Input.shape[0]):
            for j in range(Input.shape[1]):
                z_input[j] = sum(weight[j]*Input[i]) + bias[j]
                z[j]= activation_fn(z_input[j])

            y_input = sum(z*w) +b

            y = activation_fn(y_input)
            # Update the weight & bias
            if y != Target[i]:
                for j in range(weight.shape[1]):
                    weight[j]= weight[j] + lr*(Target[i]-z_input[j])*Input[i][j]
                    bias[j]  = bias[j] + lr*(Target[i]-z_input[j])

            # Store squared error value
            error.append((Target[i]-z_input)**2)
        # compute sum of square error
        Error = sum(error)
        print(k,'>> Error :',Error)
        k+=1
        
    return weight, bias

# Input dataset
x = np.array([[1.0, 1.0, 1.0], [1.0, -1.0, 1.0],
              [-1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]])
# Target values
t = np.array([1, 1, 1, -1])

w,b = Madaline(x, t, 0.0001, 3)
print('weight :',w)
print('Bias :',b)