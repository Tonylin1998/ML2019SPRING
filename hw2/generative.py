import numpy as np
import csv
from numpy.linalg import inv
import sys

    
X_train = sys.argv[3]
Y_train = sys.argv[4]
X_test = sys.argv[5]
output = sys.argv[6]

def sigmoid(x):
    a = 1/(1+np.exp(-x))
    return a





train_data_x = []
train_data_y = []

#load data
f = open(X_train, 'r', encoding='big5')
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
f = open(Y_train, 'r', encoding='big5')
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


#find mu,sigma
dim = train_data_x.shape[1]
mu1 = np.zeros(dim)
mu2 = np.zeros(dim)
size1 = 0
size2 = 0

for i in range(train_size):
    if train_data_y[i] == 1:
        mu1 += train_data_x[i]
        size1 += 1
    else:
        mu2 += train_data_x[i]
        size2 += 1
mu1 = mu1/size1
mu2 = mu2/size2

sigma1 = np.zeros((dim, dim))
sigma2 = np.zeros((dim, dim))
for i in range(train_size):
    if train_data_y[i] == 1:
        sigma1 += np.dot(np.transpose([train_data_x[i]-mu1]), [train_data_x[i]-mu1])
    else:
        sigma2 += np.dot(np.transpose([train_data_x[i]-mu2]), [train_data_x[i]-mu2])
sigma1 = sigma1/size1
sigma2 = sigma2/size2
sigma = (sigma1*size1 + sigma2*size2)/train_size
inv_sig = inv(sigma)
w = np.dot(mu1-mu2, inv_sig)
b = -(0.5)*mu1.dot(inv_sig).dot(mu1) + 0.5*mu2.dot(inv_sig).dot(mu2) + np.log(float(size1 / size2))



#load test data
f = open(X_test, 'r', encoding='big5')
lines = csv.reader(f, delimiter=",")    
test_data = []
start = 1
for line in lines:
    tmp = []
    #tmp.append( float(1) )
    if start == 0:
        for i in range(len(line)):
            tmp.append( float(line[i]) )
        test_data.append( tmp )
    if start == 1:
        start = 0          
test_data = np.array(test_data)
test_size = test_data.shape[0] 

#normalization
for i in range(test_size):
    for j in range(test_data.shape[1]):
        if std[j] != 0:
            test_data[i][j] = (test_data[i][j] - mean[j]) / std[j]
    


#predict
f_out = open(output, 'w', newline='',encoding='big5')
writer = csv.writer(f_out)
ans = []
 
#print('id,label')
for i in range(test_size):
    a = sigmoid(np.dot(w, test_data[i]) + b)
    if a > 0.5:
        v = 1
    else:
        v = 0
    #print(i+1,',',v, sep = '')
    ans.append([str(i+1)])
    ans[i].append(v)

writer.writerow(['id','label'])
for i in range(test_size):
    writer.writerow(ans[i])
f_out.close()    

    