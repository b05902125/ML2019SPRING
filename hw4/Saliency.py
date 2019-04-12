import sys
import csv
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn


class MyDataset(Dataset):
	def __init__(self, label_path):
		data = pd.read_csv(label_path)
		self.labels = np.array(data['label']).astype(int)
		tmp = data['feature'].str.split(" ").values.tolist()
		self.feature = np.array(tmp).reshape(-1,48,48,1).astype(np.float32)
	def __len__(self):
		return len(self.labels)
	def __getitem__(self, idx):
		img = self.feature[idx]
		label = self.labels[idx]
		return torch.tensor(img), torch.tensor(label.tolist())

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

if __name__ == '__main__':
	testing_data_path = sys.argv[1]
    output_file_path = sys.argv[2]

	device = torch.device('cuda')

	dataset = MyDataset(testing_data_path)
	dataloader = DataLoader(dataset, batch_size=72,  num_workers=2)

	##new_dataloader = dataloader
	path = "dri_model10.pkl?dl=1"
	model = torch.load(path)
	device = torch.device('cuda')
	model.to(device)

	print("Start")
	loss_func = nn.CrossEntropyLoss()

	count = 0

	for batch_id, batch in enumerate(dataloader):

		img, label = batch

		img_org = img.squeeze().numpy()

		##compute saliency map
		model = model.eval()
		img.requires_grad_()
		img_cuda = img.to(device)
		label_cuda = label.to(device)
		output = model(img_cuda)
		loss = loss_func(output, label_cuda)
		loss.backward()
		saliency = img.grad.abs().squeeze().data
		saliency = saliency.detach().cpu().numpy()
		predict = torch.max(output,1)[1].cpu().numpy()

		num_pics = img_org.shape[0]
		for i in range(num_pics):
			if((72*count+i) == 22):
				##plt.imsave('pic1_0.png', img_org[i], cmap=plt.cm.gray)
				plt.imsave(output_file_path + '/fig1_0.png', saliency[i], cmap=plt.cm.jet)
			elif((72*count+i) == 7192):	
				##plt.imsave('pic1_1.png', img_org[i], cmap=plt.cm.gray)
				plt.imsave(output_file_path + '/fig1_1.png', saliency[i], cmap=plt.cm.jet)
			elif((72*count+i) == 1128):	
				##plt.imsave('pic1_2.png', img_org[i], cmap=plt.cm.gray)
				plt.imsave(output_file_path + '/fig1_2.png', saliency[i], cmap=plt.cm.jet)
			elif((72*count+i) == 25):	
				##plt.imsave('pic1_3.png', img_org[i], cmap=plt.cm.gray)
				plt.imsave(output_file_path + '/fig1_3.png', saliency[i], cmap=plt.cm.jet)
			elif((72*count+i) == 6):	
				##plt.imsave('pic1_4.png', img_org[i], cmap=plt.cm.gray)
				plt.imsave(output_file_path + '/fig1_4.png', saliency[i], cmap=plt.cm.jet)
			elif((72*count+i) == 29):	
				##plt.imsave('pic1_5.png', img_org[i], cmap=plt.cm.gray)
				plt.imsave(output_file_path + '/fig1_5.png', saliency[i], cmap=plt.cm.jet)
			elif((72*count+i) == 4):	
				##plt.imsave('pic1_6.png', img_org[i], cmap=plt.cm.gray)
				plt.imsave(output_file_path + '/fig1_6.png', saliency[i], cmap=plt.cm.jet)
		count+=1


