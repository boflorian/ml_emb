import numpy as np
import matplotlib.pyplot as plt
from dnn_lib import *

learning_rate=0.075
num_iterations=10

# TODO 1: Create input and output data set. data size > 1.
# Replace None with your code.
# For example: 
# raw_x = np.array([[20, 40, 30],
#                   [45, 35, 25]])
# raw_y = np.array([[1],
#                   [0]])

# import dataset 
import pandas as pd 
df = pd.read_csv('comfort_dataset.csv')
raw_x = df[["temperature_c", "humidity_pct", "aqi"]].to_numpy()
raw_y = df[["comfortable"]].to_numpy()


INPUT_SIZE=3
HID_LAYER1=5
HID_LAYER2=4
OUTPUT_SIZE=1
np.random.seed(10)

# TODO 2: Initialize weights and bias for all connections. Weights should be initialized randomly and bias to zeros.
# use numpy library to generate random values for weight and bias, and replace None with your code.
W1 = np.random.randn(INPUT_SIZE, HID_LAYER1) * 0.01
W2 = np.random.randn(HID_LAYER1, HID_LAYER2 ) * 0.01
W3 = np.random.randn(HID_LAYER2, OUTPUT_SIZE) * 0.01

b1 = np.zeros((1, HID_LAYER1))
b2 = np.zeros((1, HID_LAYER2))
b3 = np.zeros((1, OUTPUT_SIZE))

print(raw_x)
print(raw_y)
print(raw_x.shape)

# normalize the training dataset
train_x, max_values=data_normalize(raw_x)

print("max value", max_values)
print(train_x)

cost_history = []


#train
for i in range(num_iterations):
  
    # TODO 6: Call implemented  full_forward_propagation.
    parameters = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2,
        'W3': W3,
        'b3': b3
    }

    A3, memory = full_forward_propagation(raw_x[i], parameters)
    
    # TODO 7: Call get_cost_value function 
    cost=get_cost_value(A3, raw_y[i])
    cost_history.append(cost)
    print(cost)

    #print("A1", A1)
    #print("A2", A2)
    #print("A3", A3)
    #print("Z1", Z1)
    #print("Z2", Z2)
    #print("Z3",Z3)

    # initiation of gradient descent algorithm
    dA_last = - (np.divide(raw_y.T, A3) - np.divide(1 - raw_y.T, 1 - A3))

    Z1, A1 = memory['Z1'], memory['A1']
    Z2, A2 = memory['Z2'], memory['A2']
    Z3, A3 = memory['Z3'], memory['A3']

    dA2_q, dW3_q, db3_q=single_layer_backward_propagation(dA_last, W3, b3, Z3, A2, activation="sigmoid")

    m=m = A2.shape[1]
    dZ3=A3-raw_y.T
    dW3 = np.dot(dZ3, A2.T) / m
    db3 = np.sum(dZ3, axis=1, keepdims=True) / m
    dA2 = np.dot(W3.T, dZ3)

    # TODO 10: Complete implementation of full backpropagation.
    # Output layer
    dZ3 = A3 - raw_y                        
    dW3 = (A2.T @ dZ3) / m                  
    db3 = np.sum(dZ3, axis=0, keepdims=True) / m  
    dA2 = dZ3 @ W3.T                        

    # Hidden layer 2
    dZ2 = dA2 * (Z2 > 0)                    
    dW2 = (A1.T @ dZ2) / m                  
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m  
    dA1 = dZ2 @ W2.T                        

    # Hidden layer 1
    dZ1 = dA1 * (Z1 > 0)                    
    dW1 = (raw_x.T @ dZ1) / m               
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m  


    #print("db1:", db1)
    #print("db2:", db2)

    # TODO 11: Update parameter W1, W2, W3, b1, b2, b3.
    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1


param_values={}
param_values["W1"]=W1
param_values["b1"]=b1
param_values["W2"]=W2
param_values["b2"]=b2
param_values["W3"]=W3
param_values["b3"]=b3


print("Z3:", Z3)
print("A3:", A3)

print("W3", W3)
print("b3", b3)
print("W2", W2)
print("b2", b2)
print("W1", W1)
print("b1", b1)
print(cost)


# TODO 12: Create some input data for prediction.
x_prediction=np.array([[30, 40, 90]])
x_prediction_norm=data_normalize_prediction(x_prediction, max_values)
print("x_pred_norm", x_prediction_norm)
# TODO 12: Make prediction.
A_prediction, memory=None
print("A prediction", A_prediction)

plt.plot(cost_history)
plt.show()
