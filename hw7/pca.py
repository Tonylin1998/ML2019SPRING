import numpy as np
import sys
from skimage.io import imread, imsave


image_path = sys.argv[1]
test_image = sys.argv[2]
output = sys.argv[3]
k = 5


def process(M): 
    M -= np.min(M)
    M /= np.max(M)
    M = (M * 255).astype(np.uint8)
    return M

def plot_avg_face(mean):
    average = process(mean)
    imsave('average.jpg', average.reshape(600,600,3))  
    
    
def plot_eigenface(X, k):
    for i in range(k):
        eigenface = process(X[:,i])
        imsave(str(i) + '_eigenface.jpg', eigenface.reshape(600,600,3))  
        

def reconstruct(u, mean, k):
    #for i in test_image:
    target = imread(image_path+i).flatten()
    target = target - mean
    
    weights = np.array([target.dot(u[:,i]) for i in range(k)])  
    recon_img = process(u[:,:k].dot(weights) + mean)
    
    imsave(output, recon_img.reshape(600,600,3)) 
    

def count(s, k):
    for i in range(k):
        number = s[i] * 100 / sum(s)
        print(number)




images = []
for i in range(415):
    name = image_path+'%d.jpg' % i 
    img = imread(name)
    images.append(img.flatten())
images = np.array(images)
mean = np.mean(images,axis=0)


images = images - mean

print('svd')
u, s, v = np.linalg.svd(images.T, full_matrices=False)

print(u.shape)

plot_avg_face(mean)
plot_eigenface(u, k)
reconstruct(u, mean, k)
count(s, k)
