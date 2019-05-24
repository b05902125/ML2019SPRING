import os
import sys
import numpy as np 
import sys 
from skimage.io import imread, imsave

def process(M): 
    M -= np.min(M)
    M /= np.max(M)
    M = (M * 255).astype(np.uint8)
    return M

IMAGE_PATH = sys.argv[1]

# Images for compression & reconstruction
test_image = [sys.argv[2]]
output_image = sys.argv[3]

# Number of principal components used
k = 5

##filelist = os.listdir(IMAGE_PATH) 
filelist = [f for f in os.listdir(IMAGE_PATH) if not f.startswith('.')]
print(len(filelist))

# Record the shape of images
img_shape = imread(os.path.join(IMAGE_PATH,filelist[0])).shape 

img_data = []
for filename in filelist:
    tmp = imread(os.path.join(IMAGE_PATH,filename))  
    img_data.append(tmp.flatten())

training_data = np.array(img_data).astype('float32')
print(training_data.shape)
# Calculate mean & Normalize
mean = np.mean(training_data, axis = 0)  
training_data -= mean 

# Use SVD to find the eigenvectors 
u, s, v = np.linalg.svd(training_data, full_matrices = False)  
print(u.shape, s.shape, v.shape)

print(u.shape, s.shape, v.shape)
#1.c
for x in test_image: 
    # Load image & Normalize
    picked_img = imread(os.path.join(IMAGE_PATH,x))  
    X = picked_img.flatten().astype('float32') 
    X -= mean
    
    # Compression
    weight = np.array([s.dot(u[i]) for i in range(k)])  
    # Reconstruction
    reconstruct = process(np.mean(X.reshape(1080000,1).dot(weight.reshape(1,5)), axis=1) + mean)
    imsave(output_image, reconstruct.reshape(img_shape))


#1.a
average = process(mean)
imsave('average.jpg', average.reshape(img_shape)) 

#1.b
for x in range(5):
    eigenface = process(v[x])
    imsave(str(x) + '_eigenface.jpg', eigenface.reshape(img_shape))  

#1.d
for i in range(5):
    number = s[i] * 100 / sum(s)
    print(number)