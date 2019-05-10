# standard library
import os
import csv
import sys
import argparse
import pickle
from multiprocessing import Pool
from torch.utils.data import Dataset, DataLoader

# optional library
import jieba
import pandas as pd
from gensim.models import Word2Vec

# pytorch library
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Preprocess():
    def __init__(self, data_dir, label_dir, args):
        # Load jieba library
        jieba.load_userdict(args.jieba_lib)
        self.embed_dim = args.word_dim
        self.seq_len = args.seq_len
        self.wndw_size = args.wndw
        self.word_cnt = args.cnt
        self.save_name = 'word2vec'
        self.index2word = []
        self.word2index = {}
        self.vectors = []
        self.unk = "<UNK>"
        self.pad = "<PAD>"
        # Load corpus
        if data_dir!=None:
            # Read data
            dm = pd.read_csv(data_dir)
            t_dm = pd.read_csv("test_x.csv")
            data = dm['comment']
            t_data = t_dm['comment']
            # Tokenize with multiprocessing
            # List in list out with same order
            # Multiple workers
            P = Pool(processes=4) 
            data = P.map(self.tokenize, data)
            t_data = P.map(self.tokenize, t_data)
            P.close()
            P.join()
            self.data = data
            self.t_data = t_data
            
        if label_dir!=None:
            # Read Label
            dm = pd.read_csv(label_dir)
            self.label = [int(i) for i in dm['label']]

    def tokenize(self, sentence):
        """ Use jieba to tokenize a sentence.
        Args:
            sentence (str): One string.
        Return:
            tokens (list of str): List of tokens in a sentence.
        """
        # TODO
        tokens = []
        for word in list(jieba.cut(sentence)):
        	tokens.append(word)
        return tokens

    def get_embedding(self, load=True):
        print("=== Get embedding")
        # Get Word2vec word embedding
        if load:
            embed = Word2Vec.load("new_word2vec")
            print("more training...")
            ##embed.build_vocab(self.t_data, update=True)
            ##embed.train(self.t_data, total_examples=embed.corpus_count, epochs=embed.iter)
            ##embed.save("new_word2vec")
        else:
            embed = Word2Vec(self.data, size=self.embed_dim, window=self.wndw_size, min_count=self.word_cnt, iter=16, workers=8)
            embed.save(self.save_name)
        # Create word2index dictinonary
        # Create index2word list
        # Create word vector list
        for i, word in enumerate(embed.wv.vocab):
            print('=== get words #{}'.format(i+1), end='\r')
            #e.g. self.word2index['魯'] = 1 
            #e.g. self.index2word[1] = '魯'
            #e.g. self.vectors[1] = '魯' vector
            self.word2index[word] = len(self.word2index)
            self.index2word.append(word)
            self.vectors.append(embed[word])
        self.vectors = torch.tensor(self.vectors)
        # Add special tokens
        self.add_embedding(self.pad)
        self.add_embedding(self.unk)
        print("=== total words: {}".format(len(self.vectors)))
        return self.vectors

    def add_embedding(self, word):
        # Add random uniform vector
        vector = torch.empty(1, self.embed_dim)
        torch.nn.init.uniform_(vector)
        self.word2index[word] = len(self.word2index)
        self.index2word.append(word)
        self.vectors = torch.cat([self.vectors, vector], 0)

    def get_indices(self,test=False):
        # Transform each words to indices
        # e.g. if 機器=0,學習=1,好=2,玩=3 
        # [機器,學習,好,好,玩] => [0, 1, 2, 2,3]
        all_indices = []
        # Use tokenized data
        if  os.path.isfile("trainX_100.pkl") and not test:
        	print("loading trainX_100...")
        	with open("trainX_100.pkl", "rb") as fp:
        		all_indices = pickle.load(fp)
        	return torch.LongTensor(all_indices), torch.LongTensor(self.label) 

        for i, sentence in enumerate(self.data):
            print('=== sentence count #{}'.format(i+1), end='\r')
            sentence_indices = []
            for word in sentence:
                # if word in word2index append word index into sentence_indices
                # if word not in word2index append unk index into sentence_indices
                # TODO
                if word in self.index2word:
                	sentence_indices.append(int(self.word2index[word]))
                else:
                	sentence_indices.append(int(self.word2index[self.unk]))
            # pad all sentence to fixed length
            ##print(sentence_indices)
            sentence_indices = self.pad_to_len(sentence_indices, self.seq_len, self.word2index[self.pad])
            ##print(sentence_indices)
            all_indices.append(sentence_indices)
        with open("trainX_100.pkl", "wb") as fp:
        	pickle.dump(all_indices, fp)
        if test:
            return torch.LongTensor(all_indices)         
        else:
            return torch.LongTensor(all_indices), torch.LongTensor(self.label)        

    def pad_to_len(self, arr, padded_len, padding=0):
        """ 
        if len(arr) < padded_len, pad arr to padded_len with padding.
        If len(arr) > padded_len, truncate arr to padded_len.
        Example:
            pad_to_len([1, 2, 3], 5, 0) == [1, 2, 3, 0, 0]
            pad_to_len([1, 2, 3, 4, 5, 6], 5, 0) == [1, 2, 3, 4, 5]
        Args:
            arr (list): List of int.
            padded_len (int)
            padding (int): Integer used to pad.
        Return:
            arr (list): List of int with size padded_len.
        """
        # TODO
        if(len(arr) < padded_len):
        	arr = arr + [padding for i in range(padded_len - len(arr))]
        elif(len(arr) > padded_len):
        	arr = arr[:padded_len]
       	return arr


class LSTM_Net(nn.Module):
    def __init__(self, embedding, embedding_dim, hidden_dim, num_layers, dropout=0.5, fix_emb=True):
        super(LSTM_Net, self).__init__()
        # Create embedding layer
        self.embedding = torch.nn.Embedding(embedding.size(0),embedding.size(1))
        self.embedding.weight = torch.nn.Parameter(embedding)
        # Fix/Train embedding 
        self.embedding.weight.requires_grad = False if fix_emb else True
        self.embedding_dim = embedding.size(1)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=0.5)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 6, 1024),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.Dropout(dropout),
            nn.Linear(512,1),
            nn.Sigmoid())
    def forward(self, inputs):
        inputs = self.embedding(inputs)
        x, _ = self.lstm(inputs, None)
        h1 = torch.max(x,1)[0]
        h2 = torch.mean(x,1)
        # x dimension(batch, seq_len, hidden_size)
        # Use LSTM last hidden state (maybe we can use more states)
        x = x[:, -1, :] 
        h_t = torch.cat((h1,h2,x),1)
        x = self.classifier(h_t)
        return x

def evaluation(outputs, labels):
    #outputs => probability (float)
    #labels => labels
    outputs[outputs>=0.5] = 1
    outputs[outputs<0.5] = 0
    correct = torch.sum(torch.eq(outputs, labels)).item()
    return correct

class MyDataset(Dataset):
	def __init__(self, label, data, offset, length):
		self.label = label
		self.data = data
		self.offset = offset
		self.length = length
	def __len__(self):
		return self.length
	def __getitem__(self, idx):
		inputs = self.data[idx+self.offset]
		label = self.label[idx+self.offset]
		return inputs, label

def training(args, train, valid, model, device):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\n=== start training, parameter total:{}, trainable:{}'.format(total, trainable))
    model.train()
    batch_size, n_epoch = args.batch, args.epoch
    criterion = nn.BCELoss()
    t_batch = len(train) 
    v_batch = len(valid) 
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    total_loss, total_acc, best_acc = 0, 0, 0
    for epoch in range(n_epoch):
        total_loss, total_acc = 0, 0
        # training set
        for i, (inputs, labels) in enumerate(train):
            inputs = inputs.to(device, dtype=torch.long)
            labels = labels.to(device, dtype=torch.float)
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            correct = evaluation(outputs, labels)
            total_acc += (correct / batch_size)
            total_loss += loss.item()
            print('[ Epoch{} == {}/{} ] loss:{:.3f} acc:{:.3f} '.format(
            	epoch+1, i+1, t_batch, loss.item(), correct*100/batch_size), end='\r')
        print('\nTrain | Loss:{:.5f} Acc: {:.3f} '.format(total_loss/t_batch, total_acc/t_batch*100))

        # validation set
        model.eval()
        with torch.no_grad():
            total_loss, total_acc = 0, 0
            for i, (inputs, labels) in enumerate(valid):
                inputs = inputs.to(device, dtype=torch.long)
                labels = labels.to(device, dtype=torch.float)
                outputs = model(inputs)
                outputs = outputs.squeeze()
                loss = criterion(outputs, labels)
                correct = evaluation(outputs, labels)
                total_acc += (correct / batch_size)
                total_loss += loss.item()

            print("Valid | Loss:{:.5f} Acc: {:.3f} ".format(total_loss/v_batch, total_acc/v_batch*100))
            ##if total_acc > best_acc:
            ##     best_acc = total_acc
            torch.save(model, "{}mer_ckpt_{:.3f}".format(args.model_dir,total_acc/v_batch*100))
            print('saving model with acc {:.3f}'.format(total_acc/v_batch*100))
        model.train()

def TransSplit(data, label):
	val_offset = int(len(label)*0.7)
	return MyDataset(label, data, 0, val_offset), MyDataset(label, data, val_offset, len(label)-val_offset)
	

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocess = Preprocess(args.train_X, args.train_Y, args)
    # Get word embedding vectors
    embedding = preprocess.get_embedding(load=True)
    # Get word indices
    data, label = preprocess.get_indices()
    # Split train and validation set and create data loader
    # TODO
    train_ds, val_ds = TransSplit(data, label)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=True, num_workers=4)
    # Get model
    model = LSTM_Net(embedding, args.word_dim, args.hidden_dim, args.num_layers)
    model = model.to(device)
    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)
    # Start training
    training(args, train_loader, val_loader, model, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('train_X',type=str, help='[Input] Your train_x.csv')
    parser.add_argument('jieba_lib',type=str, help='[Input] Your jieba dict.txt.big')
    parser.add_argument('train_Y',type=str, help='[Input] Your train_y.csv')

    parser.add_argument('--model_dir', default="model_100/", type=str)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--batch', default=128, type=int)
    parser.add_argument('--epoch', default=10, type=int)
    parser.add_argument('--num_layers', default=3, type=int)
    parser.add_argument('--seq_len', default=100, type=int)
    parser.add_argument('--word_dim', default=200, type=int)
    parser.add_argument('--hidden_dim', default=200, type=int)
    parser.add_argument('--wndw', default=3, type=int)
    parser.add_argument('--cnt', default=3, type=int)
    args = parser.parse_args()
    main(args)