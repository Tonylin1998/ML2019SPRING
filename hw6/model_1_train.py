from gensim.models import Word2Vec
import jieba
import pandas as pd
import numpy as np
from keras import backend as K
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, Activation, Bidirectional, Conv1D, MaxPooling1D, Reshape, Flatten
from keras.layers.recurrent import LSTM
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import sys

train_x = sys.argv[1]
train_y = sys.argv[2]
test_x = sys.argv[3]
dict_txt = sys.argv[4]


jieba.load_userdict(dict_txt)



X_train = pd.read_csv(train_x)
Y_train = pd.read_csv(train_y)
data = X_train['comment'].values
data[26073] = 'B1  台大100%>>>>>>>科大1%>>>>>>>科大100%>>>>>>>滷肉飯>>>>>>>西瓜>>>>>>>綠豆>>>>>>>你的老二>>>>>>>看三小低能兒>>>>>>>你'



X = []
for i in range(len(data)):
    tmp = []
    seg_list = jieba.cut(data[i], cut_all=True)

    for word in seg_list:
        tmp.append(word)
    X.append(tmp)
    
X = np.array(X)
Y = Y_train['label'].values
Y = np.array(Y)


X_test_data = pd.read_csv(test_x)
data_test = X_test_data['comment'].values
X_test = []
for i in range(len(data_test)):
    tmp = []
    seg_list = jieba.cut(data_test[i], cut_all=True)
    for word in seg_list:
        tmp.append(word)
    X_test.append(tmp)
    
X_test = np.array(X_test)

gensim_data = np.concatenate((X, X_test))
gensim_model = Word2Vec(gensim_data, sg=0, size=200)
gensim_model.save('gensim_1.model')


#gensim_model = Word2Vec.load('gensim_1.model')


embedding_matrix = np.zeros((len(gensim_model.wv.vocab.items()) + 1, gensim_model.vector_size))
word2idx = {}

vocab_list = [(word, gensim_model.wv[word]) for word, _ in gensim_model.wv.vocab.items()]
for i, vocab in enumerate(vocab_list):
    word, vec = vocab
    embedding_matrix[i + 1] = vec
    word2idx[word] = i + 1


def text_to_index(data_word, dictionary):
    length = []
    data_idx = []
    for line in data_word:
        tmp = []
        for word in line:
            try:
                tmp.append(dictionary[word])
            except:
                tmp.append(0)
        data_idx.append(tmp)
        length.append(len(tmp))
    return np.array(data_idx), np.max(length)


X, max_length = text_to_index(X, word2idx)
X = pad_sequences(X, maxlen=100)



model = Sequential()
model.add(Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1], 
                    weights=[embedding_matrix], trainable=True))
model.add(Dense(256, activation='relu'))
model.add(LSTM(output_dim=256, activation='tanh'))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


#checkpoint = ModelCheckpoint(filepath='model_best.h5' , monitor='val_acc', save_best_only=True, save_weights_only=False)
model.fit(X, Y, batch_size=256, epochs=2)#0, callbacks=[checkpoint], validation_split=0.1)

model.save('model_1_1.h5')




