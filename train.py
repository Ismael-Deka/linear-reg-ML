import pandas as pd
import numpy as np
import random


def MSE(y_true, y_pred):
    m = len(y_true)
    return (y_true-y_pred).pow(2).mean()*0.5

def gradient(X,  y, y_pred):
    m = len(y)
    return (X.T * (y_pred - y)).mean()

def gradient_deriv(X_i, X_j,  y, y_pred):
    return ((X_i.T * (y_pred - y))*X_j).mean()

def b_grad(y, y_pred):
    return(y_pred - y).mean()

def is_algo_stuck(list):
    result = 1
    if(len(list)==1):
        return False
    for i in range(len(list)):
        for j in range(i+1, len(list)):
            result = list[i] - list[j]
    return result == 0


df = pd.read_csv("data/data.csv")

rows = df.shape[0]

X_1_train = df['sqft_lot'].loc[:rows*0.80]
X_2_train = df['sqft_living'].loc[:rows*0.80]
X_3_train = df['bathrooms'].loc[:rows*0.80]
X_4_train = df['bedrooms'].loc[:rows*0.80]
X_5_train = df['condition'].loc[:rows*0.80]
y_train = df['price'].loc[:rows*0.80]

X_test = df['sqft_lot'].loc[rows*0.75+1:]
y_test = df['price'].loc[rows*0.75+1:]

y_train = y_train.where(y_train <= 1000000, 999999) #remove outliers

W_1 = random.uniform(-1,1)
W_2 = random.uniform(-1,1)
W_3 = random.uniform(-1,1)
W_4 = random.uniform(-1,1)
W_5 = random.uniform(-1,1)
b = random.uniform(-0.5,0.5)


y_pred = W_1 * X_1_train + W_2 * X_2_train + W_3 * X_3_train + W_4 * X_4_train+ W_5 * X_5_train+ b


cost = MSE(y_train, y_pred)
i = 1

learn_rate =  2
lr_coff =     0.01
cost_history = []
cost_history.append(cost)
momentum = 0.9

grad_one = 0
grad_two = 0
grad_three = 0
grad_4 = 0
grad_5 = 0



while(cost >= 1000):
    W_p_1 = W_1
    W_p_2 = W_2
    W_p_3 = W_3
    W_p_4 = W_4
    W_p_5 = W_5
    b_p =b
    y_pred_p = y_pred
    grad_one = gradient(X_1_train, y_train, y_pred) + (grad_one*momentum)
    grad_two = gradient_deriv(X_1_train, X_2_train, y_train, y_pred)  + (grad_two*momentum)
    grad_three = gradient_deriv(X_1_train, X_3_train, y_train, y_pred) + (grad_three*momentum)
    grad_4 = gradient_deriv(X_1_train, X_4_train, y_train, y_pred) + (grad_4*momentum)
    grad_5 = gradient_deriv(X_1_train, X_5_train, y_train, y_pred) + (grad_5*momentum)
    if(is_algo_stuck(cost_history)):
        break;
    
    W_1 = W_1-grad_one*learn_rate
    W_2 = W_2-grad_two*learn_rate
    W_3 = W_3-grad_three*learn_rate
    W_4 = W_4-grad_4*learn_rate
    W_5 = W_5-grad_5*learn_rate
    b = b-b_grad(y_train, y_pred)*learn_rate
   
    y_pred = W_1 * X_1_train + W_2 * X_2_train + W_3 * X_3_train + W_4 * X_4_train+ W_5 * X_5_train+ b
    cost = MSE(y_train, y_pred)
    if(cost > cost_history[len(cost_history)-1]):
        W_1=W_p_1
        W_2=W_p_2
        W_3=W_p_3
        W_4=W_p_4
        W_5=W_p_5
        b=b_p
        y_pred = y_pred_p
        if((learn_rate-lr_coff) <= 0):
            lr_coff/=10
        learn_rate -= lr_coff
        
        
        
        #print("learn rate updated to %.20f" % learn_rate)
        continue
    learn_rate = abs(learn_rate)
    print("Iteration: %d" % i)
    print("Learn rate: %.90f" % learn_rate)
    print("Cost: %f" % cost)
    print("Weight 1: %.20f" % W_1)
    print("Weight 2: %.20f" % W_2)
    print("Weight 3: %.20f" % W_3)
    print("Weight 4: %.20f" % W_4)
    print("Weight 5: %.20f" % W_5)
    print("Bias: %.20f" % b)
    print("---------------------------------------------")
    if(len(cost_history) == 10):
        cost_history.pop(0)
    cost_history.append(cost)
    i+=1


print("Cost: %f" % cost)
print("NRMSE: %f" % (np.sqrt(cost)/np.ptp(y_train)))
print("Mean Percent Differance: %.2f%%" % ((y_pred-y_train)/y_train*100).abs().mean())
print(y_pred)

