import pandas as pd
import numpy as np
import csv
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
from keras import backend as K
import sys


train_path = sys.argv[1]






data_train = pd.read_csv(train_path)


tmp = data_train['feature'].str.split(' ').values
X = []
for i in range(tmp.shape[0]):
    X.append(np.array(tmp[i], dtype=float))
X_train = np.array(X).reshape(-1, 48, 48, 1)

#X_train = X[:-5000, :].reshape(-1, 48, 48, 1)
#X_val = X[-5000:, :].reshape(-1, 48, 48, 1)



Y = data_train['label'].values
Y_train = np_utils.to_categorical(Y, 7)
#Y_train = Y[:-5000, :]
#Y_val = Y[-5000:, :]


X_train = X_train/255
#X_val = X_val/255







model = Sequential()


model.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=(48, 48, 1), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))

model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))

model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))

model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))

model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))

model.add(Flatten())


model.add(Dense(units=512))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Dropout(0.5))

model.add(Dense(units=512))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Dropout(0.5))

model.add(Dense(units=512))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Dropout(0.5))

model.add(Dense(units=7))
model.add(Activation('softmax'))


#name = 'model_2_best_'+str(i)+'.h5'

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#checkpoint = ModelCheckpoint(filepath='model_2_best_.h5' , monitor='val_acc', save_best_only=True, save_weights_only=False)
#early_stopping = EarlyStopping(monitor='val_loss', patience=50)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=1e-20)


#model.fit(X_train, Y_train, batch_size=128, epochs=250)

model.summary()


datagen = ImageDataGenerator(
			rotation_range=30.0,
			width_shift_range=0.2,
			height_shift_range=0.2,
			shear_range=0.2,
			zoom_range=0.2,
			horizontal_flip=True,
			#fill_mode='constant',
			#vertical_flip=False
            )
datagen.fit(X_train)

model.fit_generator(datagen.flow(X_train, Y_train, batch_size=128),
                    steps_per_epoch=X_train.shape[0]/128,
                    epochs=500,
                    #validation_data = [X_val, Y_val],
                    callbacks=[reduce_lr])



model.save('model_3.h5')
#K.clear_session()



'''
data_test = pd.read_csv('test.csv')
tmp = data_test['feature'].str.split(' ').values
X_test = []
for i in range(tmp.shape[0]):
    X_test.append(np.array(tmp[i], dtype=float))
X_test = np.array(X_test).reshape(-1, 48, 48, 1)
test_size = X_test.shape[0]

#X_test = X_test/255

ans = np.argmax(model.predict(X_test),axis=1).reshape(test_size, 1)
ans = np.concatenate((np.arange(0, test_size).reshape(test_size, 1), ans), axis=1)


f_out = open('prediction_2_nn.csv', 'w', newline='',encoding='big5')
writer = csv.writer(f_out)


writer.writerow(['id','label'])
for i in range(test_size):
    writer.writerow(ans[i])
f_out.close()    
'''
