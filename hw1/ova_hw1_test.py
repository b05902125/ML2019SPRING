import numpy as np
import scipy as sp
import pandas as panda
import csv
import sys

weight = np.load('weight7.npy')

##print(weight)

testing_data_path = sys.argv[1]
output_file_path = sys.argv[2]

fp = open(testing_data_path, 'r', encoding='big5')
rows = csv.reader(fp, delimiter=',')

count = 0
num = 0
bias = 1.0
test_x = []


for row in rows:
	if(count % 18 == 0):
		test_x.append([])
		if(count != 0):
			num+=1
	for i in range(2,11):
		if(row[i] != 'NR'):
			if(float(row[i]) >= 0.0):
				test_x[num].append(float(row[i]))
			elif(i != 2):
				test_x[num].append(float(row[i-1]))
			else:
				test_x[num].append(0.0)
		else:
			test_x[num].append(0.0)
	count+=1
##print(test_x[1])
for i in range(240):
	test_x[i].append(float(bias))

test_x_matrix = np.array(test_x, dtype='f')
answer = np.dot(test_x_matrix,weight)



wp = open(output_file_path, 'w')
writer = csv.writer(wp)
writer.writerow(['id','value'])
for i in range(240):
	tmp = 'id_'+ str(i)
	writer.writerow([tmp, int(answer[i])])
wp.close()
