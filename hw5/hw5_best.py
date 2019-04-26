import numpy as np
import torch 
import torch.nn as nn
import torchvision.transforms as transform
from torch.autograd.gradcheck import zero_gradients
from torchvision.models import vgg16, vgg19, resnet50, resnet101, densenet121, densenet169 
from PIL import Image
#import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing import image
import sys
#torch.backends.cudnn.deterministic = True
input_dir = sys.argv[1]
output_dir = sys.argv[2]

model = resnet50(pretrained=True).cuda()
model.eval()

limit = 1

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

epsilon = 0.01
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
 
    
    tmp = model(img.cuda())
    target_label = np.argmax(tmp.cpu().data.numpy())
    print(target_label)
   
    

    target = torch.Tensor([target_label])
    target = target.long()
 

    
    img_adv = img
    img_adv.requires_grad = True
    zero_gradients(img_adv)
    for j in range(100):

        if j != 0:
            img_adv = torch.Tensor(img_adv)
            img_adv = normal(img_adv)
            img_adv = img_adv.unsqueeze(0)
            img_adv.requires_grad = True
            zero_gradients(img_adv)

        output = model(img_adv.cuda())
        loss = entroy(output, target.cuda())
        loss.backward()
    
        # add epsilon to image
        img_adv = img_adv + epsilon * img_adv.grad.sign_()
    
    
       
    
        img_adv = img_adv.squeeze(0)
        img_adv = inv_normal(img_adv)
        img_adv = np.clip(img_adv.cpu().detach().numpy(), img_min, img_max)
        
        #print(img_adv)
        #img_adv = to_pil(img_adv)
    
    img_adv = np.clip(img_adv, 0, 255)
    '''
    img_try = torch.Tensor(img_adv)
    img_try = normal(img_try)
    img_try = img_try.unsqueeze(0)
    out_try = model(img_try.cuda())
    res_try = np.argmax(out_try.cpu().data.numpy())
    print(res_try) 
    if res_try==target_label:
        error.append(i)
        print('ggggggggggggg')
    '''

    
    img_adv = np.round(np.rollaxis(img_adv, 0, 3)*255)
    result = image.array_to_img(img_adv.reshape(224, 224, 3))
    result.save(output_path)
    #plt.imshow(result)
    #plt.show()



    
    
    
    
    
    
    
    
    
