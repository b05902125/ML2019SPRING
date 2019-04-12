import numpy as np
import pandas as pd
import csv
import sys
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import torchvision
import torch
import torch.nn as nn

class MyDataset(Dataset):
    def __init__(self, label_path, transform):
        data = pd.read_csv(label_path)
        self.labels = np.array(data['id']).astype(int)
        tmp = data['feature'].str.split(" ").values.tolist()
        self.feature = np.array(tmp).reshape(-1,48,48,1).astype(np.float32)
        self.transform = transform
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        img = self.feature[idx]
        img = self.transform(img)
        label = self.labels[idx]
        return img.clone().detach(), torch.tensor(label.tolist())



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



if __name__ == '__main__':

    testing_data_path = sys.argv[1]
    output_file_path = sys.argv[2]

    train_augmentation = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),torchvision.transforms.ToTensor(),torchvision.transforms.Normalize([0.486],[0.229])])
    dataset = MyDataset(testing_data_path , train_augmentation)
    dataloader = DataLoader(dataset, batch_size=24,  num_workers=2)
    predict_list = []
    label_list = []
    for i in range(1,11,1):
        new_dataloader = dataloader
        path = "dri_model"+str(i)+".pkl?dl=1"
        model = torch.load(path)
        device = torch.device('cuda')
        model.to(device)

        ##print("Start epoch")
        total_predict = np.zeros((1))
        total_label = np.zeros((1))

        for batch_id, batch in enumerate(new_dataloader):
            img, label = batch
            img_cuda = img.to(device)
            model = model.eval()
            output = model(img_cuda)
            predict = torch.max(output,1)[1]
            predict_array = predict.cpu().numpy()
            label_array = label.cpu().numpy()
            total_predict = np.append(total_predict,predict_array)
            total_label = np.append(total_label,label_array)
            
        total_predict = np.delete(total_predict, 0)
        total_label = np.delete(total_label, 0)
        predict_list.append(total_predict)
        label_list.append(total_label)

    ans_predict = np.zeros((len(label_list[0]))).astype(int)
    ans_label = label_list[0].astype(int)
    for i in range(len(label_list[0])):
        count = [0 for k in range(7)]
        for j in range(10):
            count[int(predict_list[j][i])]+=1
        ans_predict[i] = int(count.index(max(count)))

    dataframe = pd.DataFrame({'id':ans_label,'label':ans_predict})
    dataframe.to_csv(output_file_path, index=0)
