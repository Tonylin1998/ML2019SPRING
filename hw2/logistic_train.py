import numpy as np
import csv



def sigmoid(x):
    a = 1/(1+np.exp(-x))
    return a



train_data_x = []
train_data_y = []

#load data
f = open("X_train", 'r', encoding='big5')
lines = csv.reader(f, delimiter=",")
start = 1
for line in lines:
    tmp = []
    #tmp.append( float(1) )
    if start == 0:
        for i in range(len(line)):
            tmp.append( float(line[i]))
        train_data_x.append( tmp )
    if start == 1:
        start = 0      
f = open("Y_train", 'r', encoding='big5')
lines = csv.reader(f, delimiter=",")
start = 1
for line in lines:
    if start == 0:
        train_data_y.append( float(line[0]) )
    if start == 1:
        start = 0      
train_data_x = np.array(train_data_x)
train_data_y = np.array(train_data_y)    
train_size = train_data_x.shape[0]


#normalization
mean = np.mean(train_data_x, axis = 0)
std = np.std(train_data_x, axis = 0)
for i in range(train_size):
    for j in range(train_data_x.shape[1]):
        if std[j] != 0:
            train_data_x[i][j] = (train_data_x[i][j] - mean[j]) / std[j]
np.save('mean.npy', mean)
np.save('std.npy', std)

#add bias
train_data_x = np.concatenate((np.ones([train_size, 1]), train_data_x), axis=1)
print(train_data_x.shape[1])


#training
dim = train_data_x.shape[1]
w = np.zeros(dim)
pre_grad = np.zeros(dim)
x_trans = train_data_x.transpose()
learning_rate = 1
iteration = 10000
lamda = 1000
for i in range(iteration):
    y_ = sigmoid(np.dot(train_data_x, w))
    loss = y_ - train_data_y
    grad = np.dot(x_trans, loss)
    pre_grad += grad**2
    ada = np.sqrt(pre_grad)
    w -= learning_rate * grad / (ada+1e-10)
    
np.save('model_logistic.npy', w)    
    
    