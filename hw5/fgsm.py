import numpy as np
import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transform
from torch.autograd.gradcheck import zero_gradients
from PIL import Image
from torchvision.models import vgg16 , vgg19, resnet50, resnet101, densenet121, densenet169 
import pandas as pd
from torch.autograd import Variable
import sys

if __name__ == '__main__':
	testing_data_path = sys.argv[1]
	output_file_path = sys.argv[2]

	RANDOM_SEED = 8888
	torch.manual_seed(RANDOM_SEED)
	torch.cuda.manual_seed(RANDOM_SEED)
	torch.cuda.manual_seed_all(RANDOM_SEED)
	np.random.seed(RANDOM_SEED)
	##random.seed(RANDOM_SEED)



	##read label data
	data = pd.read_csv("labels.csv")
	target_label = np.array(data['TrueLabel']).reshape((200,1))
	fake1_label = target_label + 1
	fake2_label = target_label - 1
	target_label = torch.tensor(target_label)
	fake1_label = torch.tensor(fake1_label)
	fake2_label = torch.tensor(fake2_label)

	total_model = []

	epsilon = 0.08
	"""
	model = vgg16(pretrained=True)
	model.eval()
	total_model.append(model)
	
	model = vgg19(pretrained=True) ##0.725
	model.eval()
	total_model.append(model)
	"""
	model = resnet50(pretrained=True) ##0.900
	model.eval()
	total_model.append(model)
	"""
	model = resnet101(pretrained=True) ##0.800
	model.eval()
	total_model.append(model)
	
	model = densenet121(pretrained=True) ##0.815
	model.eval()
	total_model.append(model)
	
	model = densenet169(pretrained=True) ##0.820
	model.eval()
	total_model.append(model)
	"""

	criterion = nn.CrossEntropyLoss()
	normalize = transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	inv_nor = transform.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
	
	trans = transform.Compose([transform.ToTensor(), normalize])
	inverse_trans = transform.Compose([inv_nor])

	##print(target_label[1])
	for i in range(200):
		print(i)
		k = '{0:03}'.format(i)
		input_path = testing_data_path + "/" + k + ".png"
		output_path = output_file_path + "/" + k + ".png"
		im = Image.open(input_path).convert('RGB')
		image = trans(im).unsqueeze(0)
		image.requires_grad = True
		decrease = torch.zeros(image.shape)

		for j in range(len(total_model)):
			model = total_model[j]
			zero_gradients(image)
			
			output = model(image).cuda()
			loss = criterion(output, fake1_label[i].cuda()) + criterion(output, fake2_label[i].cuda()) - 4 * criterion(output, target_label[i].cuda())
			loss.backward()
			decrease += image.grad.sign_()
		image = image - epsilon * decrease.sign_()
		##print(image.shape, "before")
		image = inverse_trans(image.squeeze(0)).unsqueeze(0)
		##print(image.shape)
		torchvision.utils.save_image(image, output_path)
