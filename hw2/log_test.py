import numpy as np
import scipy as si
import pandas as pd
import csv
import sys

testing_data_path = sys.argv[5]
output_file_path = sys.argv[6]

raw_data = np.genfromtxt(testing_data_path, delimiter=',')
data_x = raw_data[1:,0:]


weight = np.load('pow_weight1.npy')

data_x = np.concatenate((data_x, np.power(data_x[:,0],2).reshape(len(data_x),1)), axis=1).astype(float)
data_x = np.concatenate((data_x, np.power(data_x[:,1],2).reshape(len(data_x),1)), axis=1).astype(float)
data_x = np.concatenate((data_x, np.power(data_x[:,5],2).reshape(len(data_x),1)), axis=1).astype(float)

mean = np.mean(data_x, axis=0)
std = np.std(data_x, axis=0)

for i in range(data_x.shape[0]):
	for j in range(data_x.shape[1]):
		if(std[j] != 0):
			data_x[i][j] = (data_x[i][j] - mean[j]) / std[j]

data_x = np.concatenate((data_x, np.ones((data_x.shape[0], 1))), axis=1).astype(float)

ans_y = 1 / (1 + np.exp(-data_x.dot(weight)))


wp = open(output_file_path, 'w')
writer = csv.writer(wp)
writer.writerow(['id','label'])
for i in range(1,len(ans_y)+1):
	if(ans_y[i-1] >= 0.5):
		writer.writerow([i, 1])
	else:
		writer.writerow([i, 0])
wp.close()
