from gensim.models import Word2Vec
import jieba
import pandas as pd
import numpy as np
from keras import backend as K
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, Activation, Bidirectional
from keras.layers.recurrent import LSTM
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
import csv
import sys


test_x = sys.argv[1]
dict_txt = sys.argv[2]
output = sys.argv[3]

jieba.load_userdict(dict_txt)



model1 = load_model('model_1_1.h5')
model2 = load_model('model_1_2.h5')
model3 = load_model('model_1_3.h5')
model4 = load_model('model_2_1.h5')
model5 = load_model('model_2_2.h5')
model6 = load_model('model_2_3.h5')
model7 = load_model('model_3_1.h5')
gensim_model = Word2Vec.load('gensim_1.model')



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

test_size = X_test.shape[0]



X_test, max_length = text_to_index(X_test, word2idx)

X_test = pad_sequences(X_test, maxlen=100)

predict1 = model1.predict(X_test)
predict2 = model2.predict(X_test)
predict3 = model3.predict(X_test)
predict4 = model4.predict(X_test)
predict5 = model5.predict(X_test)
predict6 = model6.predict(X_test)
predict7 = model7.predict(X_test)





model8 = load_model('model_no_jieba.h5')
gensim_model = Word2Vec.load('gensim_no_jieba.model')

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

test_size = X_test.shape[0]



X_test, max_length = text_to_index(X_test, word2idx)

X_test = pad_sequences(X_test, maxlen=100)

predict8 = model8.predict(X_test)














ans = predict1 + predict2 + predict3 + predict4 + predict5 + predict6 + predict7 + predict8

for i in range(test_size):
    if ans[i] >= 0.5*8:
        ans[i] = 1
    else:
        ans[i] = 0

f_out = open(output, 'w', newline='',encoding='big5')
writer = csv.writer(f_out)


writer.writerow(['id','label'])
for i in range(test_size):
    writer.writerow([ str(i), int(ans[i][0]) ])
f_out.close()    

K.clear_session()






