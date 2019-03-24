import numpy as np
import csv
import sys


X_train = sys.argv[3]
Y_train = sys.argv[4]
X_test = sys.argv[5]
output = sys.argv[6]



def sigmoid(x):
    a = 1/(1+np.exp(-x))
    return a



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
mean = np.load('mean.npy')
std = np.load('std.npy')
for i in range(test_size):
    for j in range(test_data.shape[1]):
        if std[j] != 0:
            test_data[i][j] = (test_data[i][j] - mean[j]) / std[j]
            
#add bias
test_data = np.concatenate((np.ones([test_size, 1]), test_data), axis=1)

   
  
#predict
w = np.load('model_logistic.npy')
f_out = open(output, 'w', newline='',encoding='big5')
writer = csv.writer(f_out)
ans = []
#print('id,label')
for i in range(test_size):
    a = sigmoid(np.dot(test_data[i], w))
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
    
'''
ind = np.argsort(np.abs(w))[::-1]
with open("X_test") as f:
    content = f.readline().rstrip('\n')
features = np.array([x for x in content.split(',')])
for i in ind[0:10]:
    print(features[i], w[i])
''' 