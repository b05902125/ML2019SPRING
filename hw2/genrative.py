import numpy as np
import scipy as si
import pandas as pd
import csv
import sys

X_train_path = sys.argv[3]
Y_train_path = sys.argv[4]
testing_data_path = sys.argv[5]
output_file_path = sys.argv[6]

raw_data = np.genfromtxt(X_train_path, delimiter=',')
data_x = raw_data[1:,0:]

raw_data = np.genfromtxt(Y_train_path, dtype='float')
data_y = raw_data[1:]
##data_y = data_y.reshape(len(data_y),1)


len1 = 0
data1 = np.zeros((7841,106))
len2 = 0
data2 = np.zeros((24720,106))

for i in range(len(data_y)):
	if(data_y[i] == 1):		
		data1[len1] = data_x[i]
		len1+=1
	else:
		data2[len2] = data_x[i]
		len2+=1

cov1 = pd.DataFrame(data1).cov()
cov2 = pd.DataFrame(data2).cov()
cov1_inv = np.linalg.pinv(cov1)
cov2_inv = np.linalg.pinv(cov2)
true_cov = (len1 / (len1 + len2)) * cov1 + (len2 / (len1 + len2)) * cov2
true_cov_inv = np.linalg.pinv(true_cov)

mean1 = np.mean(data1, axis=0)
mean2 = np.mean(data2, axis=0)
mean_diff = mean1 - mean2

weight = mean_diff.dot(true_cov_inv)

bias = (-0.5) * (np.transpose(mean1).dot(true_cov_inv)).dot(mean1) + 0.5 * (np.transpose(mean2).dot(true_cov_inv)).dot(mean2) + np.log(len1 / len2)


##print(weight)

raw_data = np.genfromtxt(testing_data_path, delimiter=',')
test_x = raw_data[1:,0:]

##print(test_x.dot(weight))


ans_y = 1 / (1 + np.exp(-test_x.dot(weight) - bias))


wp = open(output_file_path, 'w')
writer = csv.writer(wp)
writer.writerow(['id','label'])
for i in range(1,len(ans_y)+1):
	if(ans_y[i-1] >= 0.5):
		writer.writerow([i, 1])
	else:
		writer.writerow([i, 0])
wp.close()