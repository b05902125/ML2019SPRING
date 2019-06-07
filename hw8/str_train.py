import numpy as np
import pandas as pd
import csv
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD, Adam
import torch
import torch.nn as nn
import torchvision
import sys
import os 
import time
import pickle

def read_data(path):
	print("reading data.....")
	if os.path.isfile("feature.pkl"):
		with open("feature.pkl", "rb") as fp:
			feature = pickle.load(fp)
		with open("labels.pkl", "rb") as fp:
			labels = pickle.load(fp)	
	else:
		data = pd.read_csv(training_data_path)
		labels = np.array(data['label']).astype(int)
		tmp = data['feature'].str.split(" ").values.tolist()
		feature = np.array(tmp).reshape(-1,48,48,1).astype(np.float32)
		with open("feature.pkl", "wb") as fp:
			pickle.dump(feature, fp)
		with open("labels.pkl", "wb") as fp:
			pickle.dump(labels, fp)

	print("feature shape:", feature.shape)
	return feature, labels
	

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
		img = self.transform(img) / 255.0
		label = self.labels[idx+self.offset]
		return img.clone().detach(), torch.tensor(label.tolist())

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


def Val_Split(labels, feature):
	#train_augmentation = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
	#torchvision.transforms.RandomHorizontalFlip(),torchvision.transforms.RandomAffine(30, translate=(0.2,0.2), scale=(0.8,1.2), shear=10, resample=False, fillcolor=0),
	#torchvision.transforms.RandomRotation(5),torchvision.transforms.ToTensor(),torchvision.transforms.Normalize([0.485], [0.229])])

	train_augmentation = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
	torchvision.transforms.RandomRotation(20),
	torchvision.transforms.RandomHorizontalFlip(),torchvision.transforms.ToTensor()])
	#train_augmentation = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),torchvision.transforms.ToTensor(),torchvision.transforms.Normalize([0.485], [0.229])])
	#val_augmentation = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),torchvision.transforms.ToTensor(),torchvision.transforms.Normalize([0.485], [0.229])])
		
	val_augmentation = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),torchvision.transforms.ToTensor()])
	val_offset = int(len(labels)*0.7)

	return MyDataset(labels, feature, 0, val_offset, train_augmentation), MyDataset(labels, feature, val_offset, len(labels)-val_offset, val_augmentation)




def train(train_loader, model, criterion, optimizer):

	# switch to train mode
	model.train()
	train_acc = 0.0
	train_loss = 0.0

	for i, (inputs, target) in enumerate(train_loader):
		# measure data loading time

		target = target.cuda(async=True)

		# compute output
		output = model(inputs.cuda())
		loss = criterion(output, target)

		train_acc += np.sum(np.argmax(output.cpu().data.numpy(), axis=1) == target.cpu().data.numpy())
		train_loss += loss.item()

		# compute gradient and do SGD step
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	return train_acc, train_loss

def validate(val_loader, model, criterion):
		
		# switch to evaluate mode
		model.eval()
		val_acc = 0.0
		val_loss = 0.0
		with torch.no_grad():
			for i, (inputs, target) in enumerate(val_loader):		
				target = target.cuda(async=True)

				# compute output
				output = model(inputs.cuda())
				loss = criterion(output, target)

				val_acc += np.sum(np.argmax(output.cpu().data.numpy(), axis=1) == target.cpu().data.numpy())
				val_loss += loss.item()

		return val_acc, val_loss

def Training(train_ds, val_ds):
	model = Net()
	model.cuda()
	num_epoch = 1000
	lr = 0.001
	momentum = 0.9
	weight_decay = 1e-4
	best_acc = 0
	model_output_path = "final_model1.pth"

	optimizer = Adam(model.parameters(), lr)
	criterion = nn.CrossEntropyLoss().cuda()
	
	train_loader = DataLoader(train_ds, batch_size=24, shuffle=True, num_workers=4, pin_memory=True)
	val_loader = DataLoader(val_ds, batch_size=24, shuffle=True, num_workers=4, pin_memory=True)
	
	print("Start epoch")
	for epoch in range(num_epoch):
		epoch_start_time = time.time()
		##adjust_learning_rate(optimizer, epoch)

		# train for one epoch
		train_acc, train_loss = train(train_loader, model, criterion, optimizer)

		# evaluate on validation set
		val_acc, val_loss = validate(val_loader, model, criterion)
		val_acc = val_acc/val_ds.__len__()
		print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
		(epoch + 1, num_epoch, time.time()-epoch_start_time, train_acc/train_ds.__len__(), train_loss, val_acc, val_loss))
		
		if (val_acc > best_acc):
			#model.half()
			with open('acc.txt','w') as f:
				f.write(str(epoch)+'\t'+str(val_acc)+'\n')
			torch.save(model.state_dict(), model_output_path)
			best_acc = val_acc
			print ('Model Saved!')
			##model.float()



if __name__ == '__main__':

	training_data_path = sys.argv[1]
	feature, labels = read_data(training_data_path)
	train_ds, val_ds = Val_Split(labels, feature)
	Training(train_ds, val_ds)

	
	

