# standard library
import argparse
import csv
import time
import sys
import os
# other library
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# PyTorch library
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, total_img):
        self.total_img = total_img 
        # normalize
        self.total_img = (self.total_img ) / 255.0
        print("=== total image shape:",  self.total_img.shape)
        # shape = (40000, 3, 32, 32)


    def __len__(self):
        return len(self.total_img)

    def __getitem__(self, index):
        return(self.total_img[index])

class Net(nn.Module):
    def __init__(self, image_shape, latent_dim):
        super(Net, self).__init__()
        self.shape = image_shape
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),  #64, 16, 16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 192, 4, stride=2, padding=1),  #192, 8, 8
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.2),
            nn.Conv2d(192, 256, 4, stride=2, padding=1),  #256, 4, 4
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 384, 4, stride=2, padding=1),  #256, 2, 2
            nn.BatchNorm2d(384),
            nn.LeakyReLU(0.2),
            nn.Conv2d(384, 384, 4, stride=2, padding=1),  #384, 1, 1
            nn.BatchNorm2d(384),
            nn.LeakyReLU(0.2),
        )
        # assume output shape is (Batch, N, 1, 1)
        
        self.fc1 = nn.Sequential(
            nn.Linear(384 * 1 * 1, 300),
            nn.Linear(300, self.latent_dim),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(self.latent_dim, 300),
            nn.Linear(300, 384 * 1 * 1),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(384, 384, 4, stride=2, padding=1),  #384, 2, 2
            nn.BatchNorm2d(384),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(384, 256, 4, stride=2, padding=1),  #256, 4, 4
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 192, 4, stride=2, padding=1),  #192, 8, 8
            nn.BatchNorm2d(192),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(192, 64, 4, stride=2, padding=1),  #64, 16, 16
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),  #16, 32, 32
            nn.LeakyReLU(0.2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        # flatten
        x = x.view(len(x), -1)
        encoded = self.fc1(x)

        x = F.relu(self.fc2(encoded))
        x = x.view(-1, 384, 1, 1)
        x = self.decoder(x)
        return encoded, x
def read_test_case(path):
    dm = pd.read_csv(path)
    img1 = dm['image1_name']
    img2 = dm['image2_name']
    test_case = np.transpose(np.array([img1, img2]))
    return test_case

def clustering(model, device, loader, n_iter, reduced_dim):
    model.eval()
    latent_vec = torch.tensor([]).to(device, dtype=torch.float)
    with torch.no_grad():
        for idx, image in enumerate(loader):
            print("predict %d / %d" % (idx, len(loader)) , end='\r')
            image = image.to(device, dtype=torch.float)
            latent, r = model(image)
            latent_vec = torch.cat((latent_vec, latent), dim=0)

    latent_vec = latent_vec.cpu().detach().numpy()
    print(latent_vec.shape)
    # shape = (40000, latent_dim)

    pca = PCA(n_components="mle", copy=False, whiten=True, svd_solver='full')
    latent_vec = pca.fit_transform(latent_vec)
    print(latent_vec.shape)

    
    kmeans = KMeans(n_clusters=2, random_state=0, max_iter=n_iter).fit(latent_vec)
    return kmeans.labels_


def prediction(label, test_case, output):
    result = []
    for i in range(len(test_case)):
        index1, index2 = int(test_case[i][0])-1, int(test_case[i][1])-1
        if label[index1] != label[index2]:
            result.append(0)
        else:
            result.append(1)
    
    result = np.array(result)
    with open(output, 'w') as f:
        f.write("id,label\n")
        for i in range(len(test_case)):
            f.write("%d,%d\n" % (i, result[i]))

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    total_img = []
    iteration = 250
    reduced_dim = 100
    
    path = args.image_dir
    for i in range(1, 40001):
        print("loading image %d/40000" % i, end='\r')
        fname = os.path.join(path, "%06d.jpg" % (i))
        img = Image.open(fname)
        img.load()
        row = np.asarray(img)
        total_img.append(row)
    # since at pytorch conv layer, input=(N, C, H, W)
    total_img = np.transpose(np.array(total_img, dtype=float), (0, 3, 1, 2))
    
    test_ds = MyDataset(total_img)
    test_dl = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=4)
    # Get model
    model_path = "third.pkl"
    model = Net(total_img[0].shape, args.latent_dim)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    # read test path
    test_case = read_test_case(args.test_case)
    # Start clustering

    label = clustering(model, device, test_dl, iteration, reduced_dim)
    prediction(label, test_case, args.output_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--batch', default=128, type=int)
    #parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--latent_dim', default=128, type=int)
    parser.add_argument('image_dir', default='', type=str)
    parser.add_argument('test_case', default='', type=str)
    parser.add_argument('output_name', default='', type=str)
    ##parser.add_argument('--model_name', default='', type=str)
    args = parser.parse_args()
    main(args)