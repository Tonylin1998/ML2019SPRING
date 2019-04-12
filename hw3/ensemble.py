import pandas as pd
import numpy as np
import csv
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.models import load_model
import sys


test_path = sys.argv[1]
output_path = sys.argv[2]


model1 = load_model('model_1_0.h5')
model2 = load_model('model_1_1.h5')
model3 = load_model('model_1_2.h5')
model4 = load_model('model_2_0.h5')
model5 = load_model('model_2_1.h5')
model6 = load_model('model_2_2.h5')
model7 = load_model('model_3_0.h5')
model8 = load_model('model_3_1.h5')
model9 = load_model('model_3_2.h5')

data_test = pd.read_csv(test_path)
tmp = data_test['feature'].str.split(' ').values
X_test = []
for i in range(tmp.shape[0]):
    X_test.append(np.array(tmp[i], dtype=float))    
X_test = np.array(X_test).reshape(-1, 48, 48, 1)
test_size = X_test.shape[0]

X_test = X_test/255




predict1 = model1.predict(X_test)
predict2 = model2.predict(X_test)
predict3 = model3.predict(X_test)
predict4 = model4.predict(X_test)
predict5 = model5.predict(X_test)
predict6 = model6.predict(X_test)
predict7 = model7.predict(X_test)
predict8 = model8.predict(X_test)
predict9 = model9.predict(X_test)



p = predict1 + predict2 + predict3 + predict4 + predict5 + predict6 + predict7 + predict8 + predict9


ans = np.argmax(p, axis=1).reshape(test_size, 1)
ans = np.concatenate((np.arange(0, test_size).reshape(test_size, 1), ans), axis=1)


f_out = open(output_path, 'w', newline='',encoding='big5')
writer = csv.writer(f_out)


writer.writerow(['id','label'])

for i in range(test_size):
    writer.writerow(ans[i])
f_out.close()    




