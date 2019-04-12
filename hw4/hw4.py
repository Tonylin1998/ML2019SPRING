from skimage import color
import pandas as pd
import numpy as np
from lime import lime_image
from keras.models import load_model
from skimage.segmentation import slic
import matplotlib.pyplot as plt
from keras import backend as K
import sys


train_path = sys.argv[1]
output_path = sys.argv[2]



#load data
data_train = pd.read_csv(train_path)

tmp = data_train['feature'].str.split(' ').values
X = []
for i in range(tmp.shape[0]):
    X.append(np.array(tmp[i], dtype=float))
X_train = np.array(X).reshape(-1, 48, 48)


Y = data_train['label'].values
Y_train = Y

X_train = X_train/255

#load model
model = load_model('model_2_0.h5')

idx = [10, 8628, 17, 28, 6, 15, 4]
data = []
label = []
for i in idx:
    data.append( color.gray2rgb(X_train[i]))#.reshape(48, 48)) )
    label.append( Y_train[i] )




t = 0
for i in idx:

    pred = np.argmax(model.predict(X_train[i].reshape(-1, 48, 48, 1)))
    target = K.mean(model.output[:, pred])
    grads = K.gradients(target, model.input)[0]
    f = K.function([model.input, K.learning_phase()], [grads])

    val_grads = f([X_train[i].reshape(-1, 48, 48, 1) , 0])[0].reshape(48, 48, -1)

    val_grads *= -1
    val_grads = np.max(np.abs(val_grads), axis=-1, keepdims=True)

    # normalize
    val_grads = (val_grads - np.mean(val_grads)) / (np.std(val_grads) + 1e-5)
    val_grads *= 0.1

    
    # clip to [0, 1]
    val_grads += 0.5
    val_grads = np.clip(val_grads, 0, 1)
    


    

    heatmap = val_grads.reshape(48, 48)
    '''
    thres = 0.55
    mask = (X_train[i]*255).reshape(48, 48)
    mask[np.where(heatmap <= thres)] = np.mean(mask)
    '''

    # show original image
    plt.figure()
    plt.imshow((X_train[i]*255).reshape(48, 48), cmap='gray')
    plt.colorbar()
    plt.show()

    plt.figure()
    plt.imshow(heatmap, cmap=plt.cm.jet)
    plt.colorbar()
    path = output_path+'fig1_'+str(t)+'.jpg'
    fig = plt.gcf()
    fig.savefig(path)
    plt.show()
    

    '''
    plt.figure()
    plt.imshow(mask,cmap='gray')
    plt.colorbar()
    plt.show()
    '''
    t += 1





#### visuallize filter ####
layer_dict = dict([(layer.name, layer) for layer in model.layers])
layers = ['activation_1']
filter_num = 32
lr = 1
img = []

for layer_name in layers:
    for filter_index in range(filter_num):

        layer_output = layer_dict[layer_name].output
        loss = K.mean(layer_output[:, :, :, filter_index])        
        grads = K.gradients(loss, model.input)[0]
        grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
        f = K.function([model.input], [loss, grads])
    
    
        np.random.seed(28)
        input_img_data = np.random.random((1, 48, 48, 1))
        #print(input_img_data)
        for i in range(100):
            loss_value, grads_value = f([input_img_data])
            input_img_data += grads_value * lr
        
        img.append(input_img_data[0])
        
    fig = plt.figure(figsize=(14, 8))
    for i in range(filter_num):
        ax = fig.add_subplot(filter_num/8, 8, i+1)
        ax.imshow(img[i].reshape(48, 48), cmap='Blues')
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
        plt.tight_layout()
    fig.suptitle('filter of layer '+layer_name)
    fig.savefig(output_path+'fig2_1.jpg')
    plt.show



#### visuallize filter output ####
img_idx = 0
result = []
for layer_name in layers:
    
    f = K.function([model.input, K.learning_phase()], [layer_dict[layer_name].output])
   
    output = f([X_train[img_idx].reshape(1, 48, 48, 1), 0])[0]
    
    fig = plt.figure(figsize=(14, 8))
    for i in range(filter_num):
        ax = fig.add_subplot(filter_num/8, 8, i+1)
        ax.imshow(output[0, :, :, i], cmap='Blues')
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
        plt.tight_layout()
    fig.suptitle('output of layer '+layer_name)
    fig.savefig(output_path+'fig2_2.jpg')





#### lime ####
def predict(data):
    return model.predict(color.rgb2gray(data).reshape(-1, 48, 48, 1)).reshape(-1, 7)
    
def segmentation(data):
    return slic(data)
    
for i in range(len(idx)):
    explainer = lime_image.LimeImageExplainer()
    
    # Get the explaination of an image
    np.random.seed(16)
    #print(data[i].shape)
    explaination = explainer.explain_instance(
                                image=data[i], 
                                classifier_fn=predict,
                                segmentation_fn=segmentation
                                
                            )
    
    # Get processed image
    img, mask = explaination.get_image_and_mask(
                                    label=label[i],
                                    positive_only=False,
                                    hide_rest=False,
                                    num_features=5,
                                    min_weight=0.0
                                )
    
    # save the image
    plt.figure()
    plt.imshow(img)
    path = output_path+'fig3_'+str(i)+'.jpg'
    fig = plt.gcf()
    fig.savefig(path)
    plt.show()




