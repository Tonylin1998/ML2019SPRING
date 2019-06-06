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

train = sys.argv[1]


data_train = pd.read_csv(train)


tmp = data_train['feature'].str.split(' ').values
X = []
for i in range(tmp.shape[0]):
    X.append(np.array(tmp[i], dtype=float))
X = np.array(X)#.reshape(-1, 48, 48, 1)

X_train = X[:-5000, :].reshape(-1, 48, 48, 1)
X_val = X[-5000:, :].reshape(-1, 48, 48, 1)

print(X_train.shape)

Y = data_train['label'].values
Y = np_utils.to_categorical(Y, 7)
Y_train = Y[:-5000, :]
Y_val = Y[-5000:, :]


X_train = X_train/255
X_val = X_val/255



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



checkpoint = ModelCheckpoint(filepath='model_best_3.h5' , monitor='val_loss', save_best_only=True, save_weights_only=True)
#early_stopping = EarlyStopping(monitor='val_loss', patience=50)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, min_lr=1e-20)

#model.fit(X_train, Y_train, batch_size=128, epochs=200)#, validation_split=0.1)


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
                    epochs=600,
                    validation_data = [X_val, Y_val],
                    callbacks=[checkpoint]
                    )


arr = model.get_weights()
arr = [i.astype('float16') for i in arr]
np.save('model_3.npy',arr)






