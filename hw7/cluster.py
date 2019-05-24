import numpy as np
import pandas as pd
import csv
import keras
from keras import backend as K
from keras.preprocessing import image
from sklearn.cluster import KMeans
from keras.models import Sequential
from sklearn.decomposition import PCA
#from sklearn.manifold import TSNE
from MulticoreTSNE import MulticoreTSNE as TSNE
import pickle
from keras.models import load_model
import sys

image_path = sys.argv[1]
test_case = sys.argv[2]
output = sys.argv[3]

X_train = []
for i in range(40000):
    path = image_path+'/%06d.jpg' % (i+1)
    img = image.load_img(path, target_size=(32, 32))
    img_array = image.img_to_array(img)
    X_train.append(img_array)
X_train = np.array(X_train)
X_train /= 255
print(X_train.shape)



encoder = load_model('encoder')

tmp = encoder.predict(X_train).reshape(len(X_train), -1)

'''
pca = PCA(random_state=2) 
encoded_imgs = pca.fit_transform(tmp)
'''

tsne = TSNE(n_components=2, n_jobs=20, random_state=2)
encoded_imgs = tsne.fit_transform(tmp)


classifier = KMeans(n_clusters=2) 
classifier.fit(encoded_imgs)
labels = classifier.labels_


test_data = pd.read_csv(test_case)
image_1 = test_data['image1_name'].values
image_2 = test_data['image2_name'].values
num = len(image_1)
test_idx = []
for i in range(num):
    test_idx.append([int(image_1[i])-1, int(image_2[i])-1])
test_idx = np.array(test_idx)    
print(test_idx.shape)
print(test_idx[0])


ans = []
for i in range(num):
    if labels[test_idx[i][0]] == labels[test_idx[i][1]]:
        ans.append(1)
    else:
        ans.append(0)




f_out = open(output, 'w', newline='',encoding='big5')
writer = csv.writer(f_out)
writer.writerow(['id','label'])
for i in range(num):
    writer.writerow([ str(i), int(ans[i]) ])
f_out.close()

