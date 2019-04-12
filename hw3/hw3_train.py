import numpy as np
import pandas as pd
import csv
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import torch
import torch.nn as nn
import torchvision
import sys

class MyDataset(Dataset):
	def __init__(self, labels, feature, offset, length, transform):
		self.labels = labels
		self.feature = feature
		self.offset = offset
		self.length = length
		self.transform = transform
	def __len__(self):
		return self.length
	def __getitem__(self, idx):
		img = self.feature[idx+self.offset]
		##if(self.transform == None):
		##	img = torch.tensor(img)
		##else:
		img = self.transform(img)
		label = self.labels[idx+self.offset]
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
            nn.Dropout(0.2),
            nn.Linear(512,7),
        )
           # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x.view(-1,1,48,48))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)   
        output = self.out(x)
        return output


def trainTestSplit(labels, feature, tra_transform, val_transform):
	val_offset = int(len(labels)*0.7)
	return MyDataset(labels, feature, 0, val_offset, tra_transform), MyDataset(labels, feature, val_offset, len(labels)-val_offset, val_transform)


if __name__ == '__main__':
	device = torch.device('cuda')

	training_data_path = sys.argv[1]

	data = pd.read_csv(training_data_path)
	labels = np.array(data['label']).astype(int)
	tmp = data['feature'].str.split(" ").values.tolist()
	feature = np.array(tmp).reshape(-1,48,48,1).astype(np.float32)
	

	model = Model()
	model.to(device)
	optimizer = Adam(model.parameters(), lr=0.001)
	loss_fn = nn.CrossEntropyLoss()

	train_augmentation = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
	torchvision.transforms.RandomHorizontalFlip(),torchvision.transforms.RandomAffine(30, translate=(0.2,0.2), scale=(0.8,1.2), shear=10, resample=False, fillcolor=0),
	torchvision.transforms.RandomRotation(5),torchvision.transforms.ToTensor(),torchvision.transforms.Normalize([0.485], [0.229])])

	val_augmentation = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(), 
	torchvision.transforms.ToTensor(),torchvision.transforms.Normalize([0.485], [0.229])])

	x_label = []
	y_label = []
	train_ds, val_ds = trainTestSplit(labels, feature, train_augmentation, val_augmentation)

	
	path = "new_aug_model"
	max_acc = 0
	train_loader = DataLoader(train_ds, batch_size=24, shuffle=True, num_workers=4)
	val_loader = DataLoader(val_ds, batch_size=24, shuffle=True, num_workers=4)

	print("Start epoch")
	for epoch in range(800):
		train_loss = []
		train_acc = []
		val_acc = []
		print("epoch")
		for batch_id, batch in enumerate(train_loader):
			img, label = batch
			img_cuda = img.to(device)
			label_cuda = label.to(device)
			optimizer.zero_grad()
			model = model.train()
			output = model(img_cuda)
			loss = loss_fn(output, label_cuda)
			loss.backward()
			optimizer.step()
			predict = torch.max(output,1)[1]
			acc = np.mean((label_cuda.data == predict).cpu().numpy())
			train_acc.append(acc)
			train_loss.append(loss.item())
		print("Epoch: {}, Loss:{:.4f}, Acc: {:.4f}".format(epoch+1,np.mean(train_loss),np.mean(train_acc)))
		for batch_id, batch in enumerate(val_loader):
			img, label = batch
			img_cuda = img.to(device)
			label_cuda = label.to(device)
			model = model.eval()
			output = model(img_cuda)
			predict = torch.max(output,1)[1]
			acc = np.mean((label_cuda.data == predict).cpu().numpy())
			val_acc.append(acc)
		value_of_val = np.mean(val_acc)
		print("Validation, Epoch: {}, Acc: {:.4f}".format(epoch+1, value_of_val))
		if(value_of_val > max_acc):
			max_acc = value_of_val
			new_path = path + str(epoch) + ".pkl"
			torch.save(model, new_path)

