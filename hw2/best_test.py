import numpy as np 
import pandas as pd
import csv
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import sys


X_train = sys.argv[3]
Y_train = sys.argv[4]
X_test = sys.argv[5]
output = sys.argv[6]

mean = np.load('mean_best.npy')
std = np.load('std_best.npy')
f = open('model_best.pickle', 'rb')
model = pickle.load(f)
f.close()


more = [0,1,3,4,5]
test_data = np.genfromtxt(X_test, delimiter=',', skip_header=1, dtype='float64')
'''
org_test = np.genfromtxt("test.csv", delimiter=',', skip_header=1, dtype='int64')
en = np.array(org_test[:,4].reshape(test_data.shape[0], 1))
test_data = np.delete(test_data, dele, axis=1)
test_data = np.concatenate((en, test_data), axis=1)



fnlwgt = test_data[:,1]
new_fnl = []
max = 0
for i in range(len(fnlwgt)):
    for j in range(15):
        if fnlwgt[i]>=j*100000 and fnlwgt[i]<(j+1)*100000:
            new_fnl.append(j)
            break
test_data[:,1] = new_fnl
'''
add = []
for i in range(len(more)):
    add.append([])
    add[i] = test_data[:,more[i]].reshape(test_data.shape[0], 1)
add = np.array(add)

for i in range(len(more)):
    test_data = np.concatenate((test_data, np.sin(add[i])), axis=1)
    test_data = np.concatenate((test_data, np.cos(add[i])), axis=1)
    test_data = np.concatenate((test_data, np.tan(add[i])), axis=1)
for i in range(len(more)):
	for j in range(19):
		test_data = np.concatenate((test_data, add[i]**(j+2)), axis=1)
#test_data = test_data.astype(np.float64)
length = test_data.shape[0]

for i in range(length):
    for j in range(test_data.shape[1]):
        if std[j] != 0:
            test_data[i][j] = (test_data[i][j] - mean[j]) / std[j]
ans = model.predict(test_data).reshape(test_data.shape[0], 1).astype(np.int32)
ans = np.concatenate((np.arange(1, length+1).reshape(length, 1), ans), axis=1)


f_out = open(output, 'w', newline='',encoding='big5')
writer = csv.writer(f_out)
writer.writerow(['id','label'])
for i in range(length):
    writer.writerow(ans[i])
f_out.close() 

