# standard library
import os
import csv
import sys
import argparse
import pickle
import numpy as np
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
            data = dm['comment']
            # Tokenize with multiprocessing
            # List in list out with same order
            # Multiple workers
            P = Pool(processes=4) 
            data = P.map(self.tokenize, data)
            P.close()
            P.join()
            self.data = data
            
        

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
            print("loading_word2vec")
            embed = Word2Vec.load("new_word2vec?dl=1%0D")
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

    def get_indices(self,test=True):
        # Transform each words to indices
        # e.g. if 機器=0,學習=1,好=2,玩=3 
        # [機器,學習,好,好,玩] => [0, 1, 2, 2,3]
        all_indices = []
        # Use tokenized data
        if  os.path.isfile("testX_100.pkl"):
            print("loading testX...")
            with open("testX_100.pkl", "rb") as fp:
                all_indices = pickle.load(fp)
            ##for i in range(14480, 14497):
            ##    print(i, all_indices[i])
            ##print(self.index2word[44765], self.index2word[44766])
            return torch.LongTensor(all_indices)

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
        with open("testX_100.pkl", "wb") as fp:
            pickle.dump(all_indices, fp)
        if test:
            return torch.LongTensor(all_indices)            

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
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 1024),
            nn.Dropout(dropout),
            nn.Linear(1024, 1),
            nn.Sigmoid())
    def forward(self, inputs):
        inputs = self.embedding(inputs)
        x, _ = self.lstm(inputs, None)
        # x dimension(batch, seq_len, hidden_size)
        # Use LSTM last hidden state (maybe we can use more states)
        x = x[:, -1, :] 
        x = self.classifier(x)
        return x

def evaluation(outputs):
    #outputs => probability (float)
    #labels => labels
    outputs[outputs>=0.5] = 1
    outputs[outputs<0.5] = 0
    return outputs

class MyDataset(Dataset):
    def __init__(self, data, length):
        self.data = data
        self.length = length
    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        inputs = self.data[idx]
        return inputs

def testing(args, test, model, device):
    model.eval()
    total_predict = []
    with torch.no_grad():
        for i, inputs in enumerate(test):
            inputs = inputs.to(device, dtype=torch.long)
            outputs = model(inputs)
            outputs = outputs.squeeze()
            ##correct = evaluation(outputs)
            ans = outputs.cpu().numpy()
            ans = ans.reshape(len(ans)).tolist()
            total_predict += ans
    return total_predict

        
    


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocess = Preprocess(args.test_X, args.test_Y, args)
    # Get word embedding vectors
    embedding = preprocess.get_embedding(load=True)
    # Get word indices
    data = preprocess.get_indices(test=True)
    # Split train and validation set and create data loader
    # TODO
    print(data.shape)
    test_ds = MyDataset(data, len(data))
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=4)
    # Get model
    ##model = LSTM_Net(embedding, args.word_dim, args.hidden_dim, args.num_layers)
    model_name = ["new_ckpt_75.266?dl=1%0D", "new_ckpt_75.072?dl=1%0D", "drp_ckpt_75.116?dl=1%0D", "drp_ckpt_74.842?dl=1%0D"]
    answer = np.zeros((len(data)))
    for name in model_name:
        path = name
        model = torch.load(path)
        model = model.to(device)
        # Start training
        predict = testing(args, test_loader, model, device)
        ##predict = [int(i) for i in predict]
        answer += np.array(predict)
    answer = answer / len(model_name)
    answer = evaluation(answer)
    answer = answer.astype(np.uint8)
    idx = np.arange(20000).astype(int)
    test_df = pd.DataFrame({'id': idx, 'label': answer})
    test_df.to_csv(args.test_Y, index=0)

        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('test_X',type=str, help='[Input] Your test_x.csv')
    parser.add_argument('jieba_lib',type=str, help='[Input] Your jieba dict.txt.big')
    parser.add_argument('test_Y',type=str, help='[Output] Your test_y.csv')

    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--batch', default=128, type=int)
    parser.add_argument('--epoch', default=30, type=int)
    parser.add_argument('--num_layers', default=5, type=int)
    parser.add_argument('--seq_len', default=100, type=int)
    parser.add_argument('--word_dim', default=200, type=int)
    parser.add_argument('--hidden_dim', default=200, type=int)
    parser.add_argument('--wndw', default=3, type=int)
    parser.add_argument('--cnt', default=3, type=int)
    args = parser.parse_args()
    main(args)