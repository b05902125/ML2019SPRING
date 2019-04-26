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


	epsilon = 0.2
	alpha = 0.01

	model = resnet50(pretrained=True) ##0.900
	model.eval()


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
		##zero_gradients(image)
		image_variable = torch.zeros(image.shape)
		image_variable.data = image
		image_variable.requires_grad = True
		for step in range(5):
			zero_gradients(image_variable)
			output = model.forward(image_variable).cuda()
			loss = criterion(output, fake1_label[i].cuda()) + criterion(output, fake2_label[i].cuda()) - criterion(output, target_label[i].cuda())
			loss.backward()
			x_grad = alpha * torch.sign(image_variable.grad.data)
			adv_temp = image_variable.data - x_grad
			total_grad = adv_temp - image
			total_grad = torch.clamp(total_grad, -epsilon, epsilon)
			x_adv = image + total_grad
			image_variable.data = x_adv
			
		ans_image = inverse_trans(image_variable.data.squeeze(0)).unsqueeze(0)
		torchvision.utils.save_image(ans_image, output_path)
