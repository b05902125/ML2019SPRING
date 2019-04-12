import sys
import csv
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn


testing_data_path = sys.argv[1]
output_file_path = sys.argv[2]

data = pd.read_csv(testing_data_path)
labels = np.array(data['label']).astype(int)
tmp = data['feature'].str.split(" ").values.tolist()
feature = np.array(tmp).reshape(-1,48,48,1).astype(np.float32)
img = torch.tensor(feature)
label = torch.tensor(labels)
torch.save(img,"train_data.pth")
torch.save(label, "train_label.pth")