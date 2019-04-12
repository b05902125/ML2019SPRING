import numpy as np
import torch
from lime import lime_image
from skimage.segmentation import slic
import matplotlib.pyplot as plt
import torch.nn as nn
import sys

class Model(nn.Module):
    ##(W — F + 1) / S
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Sequential( 
            nn.Conv2d(
                in_channels=1,      # input height
                out_channels=32,    # n_filters
                kernel_size=3,      # filter size 7
                stride=1,           # filter movement/step 2
                padding=1,          # 3
            ),      
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),    
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),  
            nn.LeakyReLU(0.2), 
            nn.MaxPool2d(2),
            nn.Dropout(0.1),

            nn.Conv2d(64, 192, 3, 1, 1), 
            nn.BatchNorm2d(192),  
            nn.LeakyReLU(0.2),  
            nn.MaxPool2d(2),
            nn.Dropout(0.1),

            nn.Conv2d(192, 384, 3, 1, 1), 
            nn.BatchNorm2d(384),  
            nn.LeakyReLU(0.2),  
            nn.MaxPool2d(2),
            nn.Dropout(0.1),

            nn.Conv2d(384, 256, 3, 1, 1), 
            nn.BatchNorm2d(256),  
            nn.LeakyReLU(0.2),  
            ##nn.MaxPool2d(2), 
        )
        self.avgpool = nn.AdaptiveAvgPool2d((3, 3))
        self.out = nn.Sequential(
            nn.Linear(256 * 3 * 3, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(512,512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(512,7),
        )
           # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x.view(-1,1,48,48))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)   
        output = self.out(x)
        return output



# two functions that lime image explainer requires
def predict(image):
    model = torch.load("dri_model10.pkl?dl=1")
    model = model.eval().cuda()
    image_tp = image[:,:,:,0].astype('float32')

    ##print(image_tp.shape)
    new_image = torch.tensor(image_tp)
    output = model(new_image.cuda())
    predict_array = output.cpu().detach().numpy()
    return predict_array

def segmentation(image):
    ##print("Seg")
    return slic(image, n_segments=100)

if __name__ == '__main__':
    testing_data_path = sys.argv[1]
    output_file_path = sys.argv[2]
    # load data and model
    x_train = torch.load('train_data.pth')
    x_label = torch.load('train_label.pth')

    x_train_rgb = torch.stack([x_train, x_train, x_train], 3)
    x_train_rgb = torch.squeeze(x_train_rgb, dim=4)


    model = torch.load("dri_model10.pkl?dl=1")
    model.eval().cuda()

    explainer = lime_image.LimeImageExplainer()

    idx_array = [22, 7192, 1128, 25, 6, 29, 4]

    for i in range(7):
        new_train_rgb = x_train_rgb.cpu().numpy().astype(int)
        np.random.seed(8888)
        idx = idx_array[i]
        explaination = explainer.explain_instance(image=new_train_rgb[idx], classifier_fn=predict,segmentation_fn=segmentation)
        new_x_label = x_label.cpu().numpy().astype(int)
        image, mask = explaination.get_image_and_mask(
                                        label=new_x_label[idx],
                                        positive_only=False,
                                        hide_rest=False,
                                        num_features=5,
                                        min_weight=0.0)

        plt.imsave(output_file_path + '/fig3_' + str(i) + '.jpg', image.astype(np.uint8))
