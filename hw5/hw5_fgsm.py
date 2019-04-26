import numpy as np
import torch 
import torch.nn as nn
import torchvision.transforms as transform
from torch.autograd.gradcheck import zero_gradients
from torchvision.models import vgg16, vgg19, resnet50, resnet101, densenet121, densenet169 
#import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from keras.preprocessing import image
import sys

input_dir = sys.argv[1]
output_dir = sys.argv[2]


model = resnet50(pretrained=True)
model.eval()

limit = 3

normal = transform.Compose([transform.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])
                            ])

inv_normal = transform.Compose([transform.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transform.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ])
                               ])

to_tensor = transform.Compose([transform.ToTensor()])
to_pil = transform.Compose([transform.ToPILImage()])


entroy = nn.CrossEntropyLoss()

epsilon = 0.04
error = []


for i in range(200):
    if i<10:
        name = '/00'+str(i)+'.png'
    elif i>=10 and i<100:
        name = '/0'+str(i)+'.png'    
    else:
        name = '/'+str(i)+'.png'
    input_path = input_dir+name
    output_path = output_dir+name
    img_org = Image.open(input_path)
    img_array = np.array(img_org)


    img_t = to_tensor(img_org)
    img = normal(img_t)
    img = img.unsqueeze(0)
    img.requires_grad = True

    img_min = img_t.numpy()-limit/255

    
    img_max = img_t.numpy()+limit/255

    
    
    output = model(img)
    target_label = np.argmax(output.data.numpy())
    print(target_label)
   
    
    #target = torch.Tensor(target)
    target = torch.Tensor([target_label])
    target = target.long()
    # set gradients to zero
    zero_gradients(img)


    
    loss = entroy(output, target)
    #model.zero_grad()
    loss.backward()
 


    # add epsilon to image
    img_adv = img + epsilon * img.grad.sign_()


    #print(img.grad.sign_())
    

    #img_adv = torch.Tensor(img_adv)

    
    


    '''
    out_try = model(img_adv)
    res_try = np.argmax(out_try.data.numpy())
    print(res_try)
    print(i) 
    if res_try==target_label:
        error.append(i)
    '''

    img_adv = img_adv.squeeze(0)
    img_adv = inv_normal(img_adv)
    img_adv = np.clip(img_adv.detach().numpy(), img_min, img_max)
    #print(img_adv)
    #img_adv = img_adv.detach().numpy()
    img_adv = np.rollaxis(img_adv, 0, 3)*255
    img_adv = np.clip(img_adv, 0, 255)
    #print(img_adv)
    #img_adv = to_pil(img_adv)
    
    
    #img_new = np.array(img_adv, dtype=int)
    #print(img_new, img_array)
    #print(img_new-img_array)
    result = image.array_to_img(img_adv.reshape(224, 224, 3))
    result.save(output_path)
    #plt.imshow(result)
    #plt.show()



    
    
    
    
    
    
    
    
    