import numpy as np
import sys
import torch
from torch.optim import Adam
import torch.nn as nn
import torchvision 
import matplotlib.pyplot as plt

##from misc_functions import preprocess_image, recreate_image, save_image


class Model(nn.Module):
    ##(W — F + 1) / S
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Sequential( 
            nn.Conv2d(1,32,3,1,1),      
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

    def __getitem__(self,index):
        if(index == 0):
            return self.conv1



class CNNLayerVisualization():
    """
        Produces an image that minimizes the loss of a convolution
        operation for a specific layer and filter
    """
    def __init__(self, model, selected_layer, selected_filter):
        self.model = model
        self.model.eval().cuda()
        self.selected_layer = selected_layer
        self.selected_filter = selected_filter
        self.conv_output = 0
        # Create the folder to export images if not exists

    
    def hook_layer(self):
        def hook_function(module, grad_in, grad_out):
            # Gets the conv output of the selected filter (from selected layer)
            self.conv_output = grad_out[0, self.selected_filter]
        # Hook the selected layer
        self.model[0][self.selected_layer].register_forward_hook(hook_function)
    

    def visualise_layer_with_hooks(self, output_file_path):
        # Hook the selected layer
        self.hook_layer()
        # Generate a random image
        np.random.seed(8888)
        random_image = np.uint8(np.random.uniform(150, 180, (1, 48, 48)))
        # Process image and return variable
        processed_image = torch.tensor(random_image).type('torch.FloatTensor')
        processed_image.requires_grad = True
        # Define optimizer for the image
        optimizer = Adam([processed_image],lr=0.1)
        for i in range(1000):
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            x = processed_image.cuda().view(-1,1,48,48)
            ##print(processed_image)
        
            ##conv_output = self.model[0][0](x.view(-1,1,48,48))
            ##for j in range(1,10):
            ##	conv_output = self.model[0][j](conv_output)

            for index, layer in enumerate(self.model[0]):
            	x = layer(x)
            	if(index == self.selected_layer):
            		break


            loss = -torch.mean(self.conv_output)
            ##print(self.conv_output.shape)
            ##print(processed_image.shape)
            ##print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.cpu().numpy()))
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            ##self.created_image = recreate_image(processed_image)
            # Save image
        print("Loss:", loss, "filter:", self.selected_filter)
        im_path = output_file_path + "/fig2_1.jpg"
        new_image = processed_image.detach().numpy().reshape((48,48)).astype(np.uint8)
        plt.imsave(im_path, new_image, cmap=plt.cm.pink)


class PICLayerVisualization():
    """
        Produces an image that minimizes the loss of a convolution
        operation for a specific layer and filter
    """
    def __init__(self, model, selected_layer, selected_filter, image):
        self.model = model
        self.model.eval().cuda()
        self.selected_layer = selected_layer
        self.selected_filter = selected_filter
        self.conv_output = 0
        self.image = image
        # Create the folder to export images if not exists

    
    def hook_layer(self):
        def hook_function(module, grad_in, grad_out):
            # Gets the conv output of the selected filter (from selected layer)
            self.conv_output = grad_out[0, self.selected_filter]
        # Hook the selected layer
        self.model[0][self.selected_layer].register_forward_hook(hook_function)
    

    def visualise_layer_with_hooks(self, output_file_path):
        # Hook the selected layer
        self.hook_layer()
        # Generate a random image
        np.random.seed(8888)
        processed_image = self.image.type('torch.FloatTensor')
        
        x = processed_image.cuda().view(-1,1,48,48)
        ##print(processed_image)
    
        for index, layer in enumerate(self.model[0]):
        	x = layer(x)
        	if(index == self.selected_layer):
        		break

       	x = x.squeeze_(dim=0)
        print(x.shape)
        print("filter:", self.selected_filter)
        ##for i in range(64):
        	##print(i)
        im_path = output_file_path + "/fig2_2.jpg"
        new_image =x[37].cpu().detach().numpy().astype(np.uint8)
        plt.imsave(im_path, new_image, cmap=plt.cm.pink)




if __name__ == '__main__':

	testing_data_path = sys.argv[1]
    output_file_path = sys.argv[2]

    cnn_layer = 0
    filter_pos = 10
    # Fully connected layer is not needed
    x_train = torch.load('train_data.pth')
    x_label = torch.load('train_label.pth')

    pretrained_model = torch.load("dri_model10.pkl?dl=1")
    ##for i in range(128):
    ##	filter_pos = i
    ##	layer_vis = CNNLayerVisualization(pretrained_model, 14, filter_pos)
    ##	layer_vis.visualise_layer_with_hooks()
    ##for i in range(64):
    ##	filter_pos = i
    ##	pic_vis = PICLayerVisualization(pretrained_model, 0, filter_pos, x_train[0])
    ##	pic_vis.visualise_layer_with_hooks()

    layer_vis = CNNLayerVisualization(pretrained_model, 14, 95)
    layer_vis.visualise_layer_with_hooks(output_file_path)
    pic_vis = PICLayerVisualization(pretrained_model, 0, 1, x_train[0])
    pic_vis.visualise_layer_with_hooks(output_file_path)
	   	