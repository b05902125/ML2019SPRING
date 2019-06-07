import numpy as np
import pandas as pd
import csv
import os
import sys
import pickle
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import torchvision
import torch
import torch.nn as nn

class MyDataset(Dataset):
    def __init__(self, feature, transform):
        self.feature = feature
        self.transform = transform
    def __len__(self):
        return len(self.feature)
    def __getitem__(self, idx):
        img = self.feature[idx]
        img = self.transform(img) / 255.0
        return img.clone().detach()

def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Conv') == 0:
        m.weight.data.normal_(0.0, 0.02)

class Net(nn.Module):
#output = (input - kernel + 2 * padding) / stride + 1
    def __init__(self):
        super(Net, self).__init__()
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
            #nn.Dropout(0.1),

            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),  
            nn.LeakyReLU(0.2), 
            nn.MaxPool2d(2),
            #nn.Dropout(0.1),
            
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),  
            nn.LeakyReLU(0.2),
            #nn.Dropout(0.1), 

            nn.Conv2d(32, 60, 3, 1, 1), 
            nn.BatchNorm2d(60),  
            nn.LeakyReLU(0.2),  
            nn.MaxPool2d(2),
            
        )
        #self.avgpool = nn.AdaptiveAvgPool2d((3, 3))
        self.out = nn.Sequential(
            nn.Linear(60 * 6 * 6, 7),
            nn.Sigmoid()
        )
        self.conv1.apply(gaussian_weights_init)
        self.out.apply(gaussian_weights_init)
# fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x.view(-1,1,48,48))
        #x = self.avgpool(x)
        x = x.view(x.size(0), -1)   
        output = self.out(x)
        return output

def handle_data(path):
    print("handle_data....")
    if os.path.isfile("test.pkl"):
        with open("test.pkl", "rb") as fp:
            feature = pickle.load(fp)
    else:
        data = pd.read_csv(path)
        labels = np.array(data['id']).astype(int)
        tmp = data['feature'].str.split(" ").values.tolist()
        feature = np.array(tmp).reshape(-1,48,48,1).astype(np.float32)
        with open("test.pkl", "wb") as fp:
            pickle.dump(feature, fp)
    train_augmentation = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),torchvision.transforms.ToTensor()])
    test_ds = MyDataset(feature, train_augmentation)
    return test_ds

def Testing(dataloader, output_file_path):

    for i in range(0,2,1):
        path = "save/half_final_model" + str(i+1) +".pth"
        model = Net()
        model.load_state_dict(torch.load(path))
        model.cuda()
        model.float()
        model.eval()
        total_predict = np.zeros((1,7))

        ##print("Start epoch")
        
        print("Start")
        with torch.no_grad():
            for batch_id, img in enumerate(dataloader):
                output = model(img.cuda())
                predict_array = output.cpu().numpy()
                total_predict = np.concatenate((total_predict,predict_array))
        total_predict = np.delete(total_predict, 0, axis=0)
        if(i == 0):
            ans_predict = total_predict
        else:
            ans_predict = ans_predict + total_predict

    print(ans_predict.shape)
    length = int(len(ans_predict))
    id_list = np.arange(0,length)
    print(id_list, length)
    
    final_predict = np.argmax(ans_predict, axis=1).astype(np.uint8)
    dataframe = pd.DataFrame({'id':id_list,'label':final_predict})
    dataframe.to_csv(output_file_path, index=0)

    

if __name__ == '__main__':

    testing_data_path = sys.argv[1]
    output_file_path = sys.argv[2]

    test_ds = handle_data(testing_data_path)
    dataloader = DataLoader(test_ds, batch_size=24,  num_workers=2)
    Testing(dataloader, output_file_path)
    