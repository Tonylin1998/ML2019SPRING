import numpy as np 
import pandas as pd
import csv
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


train_data_x = np.genfromtxt("X_train", delimiter=',', skip_header=1, dtype='float64')
train_data_y = np.genfromtxt("Y_train", delimiter=',', skip_header=1, dtype='float64')

#train_data_x = pd.read_csv('X_train').values
#train_data_y = pd.read_csv('Y_train').values

'''
org_train = np.genfromtxt("train.csv", delimiter=',', skip_header=1, dtype='float64')
en = np.array(org_train[:,4].reshape(train_data_x.shape[0], 1))
dele = [15,16,17,18,19,20,21]
train_data_x = np.delete(train_data_x, dele, axis=1)
train_data_x = np.concatenate((en, train_data_x), axis=1)
'''
'''
fnlwgt = train_data_x[:,1]
new_fnl = []
max = 0
for i in range(len(fnlwgt)):
    for j in range(15):
        if fnlwgt[i]>=j*100000 and fnlwgt[i]<(j+1)*100000:
            new_fnl.append(j)
            break
train_data_x[:,1] = new_fnl
print(train_data_x[:,1])
'''
more = [0,1,3,4,5]

add = []
for i in range(len(more)):
    add.append([])
    add[i] = train_data_x[:,more[i]].reshape(train_data_x.shape[0], 1)
add = np.array(add)


for i in range(len(more)):
    train_data_x = np.concatenate((train_data_x, np.sin(add[i])), axis=1)
    train_data_x = np.concatenate((train_data_x, np.cos(add[i])), axis=1)
    train_data_x = np.concatenate((train_data_x, np.tan(add[i])), axis=1)
    #train_data_x = np.concatenate((math.sin(add[i]), train_data_x), axis=1)
    
    
for i in range(len(more)):
	for j in range(19):
		train_data_x = np.concatenate((train_data_x, add[i]**(j+2)), axis=1)



#train_data_x = train_data_x.astype(np.float64)


mean = np.mean(train_data_x, axis = 0)
std = np.std(train_data_x, axis = 0)
for i in range(train_data_x.shape[0]):
    for j in range(train_data_x.shape[1]):
        if std[j] != 0:
            train_data_x[i][j] = (train_data_x[i][j] - mean[j]) / std[j]
np.save('mean_best.npy', mean)
np.save('std_best.npy', std)

            
           
#max_depth=5, n_estimators = 1000
model = GradientBoostingClassifier(max_depth=3, n_estimators = 1000)

#model = SVC()

#solver='lbfgs', C=1000, max_iter=5000
#model = LogisticRegression(solver='lbfgs')


#scores = cross_val_score(model, train_data_x, train_data_y, cv=5, scoring='accuracy')
#print(scores, scores.mean())
model.fit(train_data_x, train_data_y.ravel())


f = open('model_best.pickle', 'wb')
pickle.dump(model, f)
f.close()














