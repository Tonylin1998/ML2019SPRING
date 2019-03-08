import numpy as np
import csv
import math

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
            elif float(line[i]) < 0:
                if i == 0:
                    data[k].append( float(0) )
                    pre = float(0)
                else:
                    data[k].append( pre )
            else:
                data[k].append( float(line[i]) )
                pre = float(line[i])
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
            if j == 14 and j == 15:
                for k in range(9):
                    tmp.append( data[j][480*m+i+k]*math.pi/180 )
            else:
                for k in range(9):
                    tmp.append( data[j][480*m+i+k] )
        for k in range(9):
            tmp.append( data[9][480*m+i+k]**2 )
        for k in range(9):
            tmp.append( data[8][480*m+i+k]**2 )
        train_data_x.append( tmp )
        train_data_y.append( data[9][480*m+i+9] )
    
train_data_x = np.array(train_data_x)
train_data_y = np.array(train_data_y)    
#print(train_data_x.shape[1])

dim = train_data_x.shape[1]
pre_grad = np.zeros(dim)
w = np.zeros(dim)
m = np.zeros(dim)
v = np.zeros(dim) 
b_1 = 0.9
b_2 = 0.999
x_trans = train_data_x.transpose()
x_trans_y = np.dot(x_trans, train_data_y)
x_trans_x = np.dot(x_trans, train_data_x)

learning_rate = 1
iteration = 1000000


for i in range(iteration):

    grad = 2*( np.dot(x_trans_x, w) - x_trans_y )
    '''
    m = b_1*m + (1-b_1)*grad
    v = b_2*v + (1-b_2)*grad*grad
    m_hat = m/(1-(b_1**(i+1)))  
    v_hat = v/(1-(b_2**(i+1)))
    w -= learning_rate * m_hat / np.sqrt(v_hat)
    '''
    pre_grad += grad**2
    ada = np.sqrt(pre_grad)
    w -= learning_rate * grad / ada

np.save('model_best.npy', w)
    
    