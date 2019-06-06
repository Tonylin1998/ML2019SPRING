import pandas as pd
import numpy as np
import csv
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization, SeparableConv2D, AveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K
import sys

test = sys.argv[1]
output = sys.argv[2]




model = Sequential()



model.add(SeparableConv2D(filters=128, kernel_size=(3, 3), input_shape=(48, 48, 1), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(SeparableConv2D(filters=128, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(SeparableConv2D(filters=64, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(SeparableConv2D(filters=64, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(SeparableConv2D(filters=64, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(SeparableConv2D(filters=64, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())
model.add(Dense(units=7))
model.add(Activation('softmax'))



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()



arr = np.load('model_best_3.npy')
arr = [i.astype('float32') for i in arr]
model.set_weights(arr)
'''

model.load_weights('model_best_3.h5')
arr = model.get_weights()
arr = [i.astype('float16') for i in arr]
np.save('model_best_3.npy',arr)
'''


data_test = pd.read_csv(test)
tmp = data_test['feature'].str.split(' ').values
X_test = []
for i in range(tmp.shape[0]):
    X_test.append(np.array(tmp[i], dtype=float))
X_test = np.array(X_test).reshape(-1, 48, 48, 1)
test_size = X_test.shape[0]

X_test = X_test/255

ans = np.argmax(model.predict(X_test),axis=1).reshape(test_size, 1)
ans = np.concatenate((np.arange(0, test_size).reshape(test_size, 1), ans), axis=1)


f_out = open(output, 'w', newline='',encoding='big5')
writer = csv.writer(f_out)


writer.writerow(['id','label'])
for i in range(test_size):
    writer.writerow(ans[i])
f_out.close()    













