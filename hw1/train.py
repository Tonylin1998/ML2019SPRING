import numpy as np
import csv

f = open("train.csv", 'r', encoding='big5')
lines = csv.reader(f, delimiter=",")

data = []
for i in range(18):
	data.append([])

k = 0
start = 1
for line in lines:
    if start == 0:
        for i in range(3, 27):
            if line[i] == "NR":
                    data[k].append( float(0) )
            else:
                data[k].append( float(line[i]) )
        k += 1
        if k == 18:
            k = 0
    if start == 1:
        start = 0      
train_data_x = []
train_data_y = []

data = np.array(data)
length = 20*24 - 9

for m in range(12):
    for i in range(length):
        tmp = []
        tmp.append( float(1) )
        for j in range(18):
            for k in range(9):
                tmp.append( data[j][480*m+i+k] )
        train_data_x.append( tmp )
        train_data_y.append( data[9][480*m+i+9] )
    
train_data_x = np.array(train_data_x)
train_data_y = np.array(train_data_y)    
#print(train_data_x.shape[1])

dim = train_data_x.shape[1]
w = np.zeros(dim)
pre_grad = np.zeros(dim)
x_trans = train_data_x.transpose()
x_trans_y = np.dot(x_trans, train_data_y)
x_trans_x = np.dot(x_trans, train_data_x)


learning_rate = 10
iteration = 100000


for i in range(iteration):
    '''
    y_ = np.dot(train_data_x, w)
    loss = y_ - train_data_y
    grad = 2*np.dot(x_trans, loss)
    '''
    grad = 2*( np.dot(x_trans_x, w) - x_trans_y ) 
    pre_grad += grad**2
    ada = np.sqrt(pre_grad)
    w -= learning_rate * grad / ada

np.save('model.npy', w)
    




    
    
    
    
    
    
    
    
    
    