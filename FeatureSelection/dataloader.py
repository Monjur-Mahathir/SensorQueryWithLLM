import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

class UCIHARDataset(Dataset):
    def __init__(self, feature_file, label_file, feature_indices):
        self.Xs, self.ys = self.process_file(feature_file, label_file, feature_indices)
    
    def process_file(self, feature_file, label_file, feature_indices):
        #train_file = "E:/LLM/Llama_Intro/UCI HAR Dataset/train/y_train.txt"  
        Xs, ys = [], []
        
        with open(feature_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                data = line.split()
                data = np.array([float(i) for i in data])
                data = data[feature_indices]
                Xs.append(data)
        f.close()
        Xs = np.array(Xs)

        with open(label_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                label = int(line.split()[0]) - 1
                ys.append(label)
        f.close()
        ys = np.array(ys)

        return Xs, ys
        
    def __len__(self):
        return self.Xs.shape[0]
    
    def __getitem__(self, idx):
        X = self.Xs[idx]
        X = torch.tensor(X)

        y = self.ys[idx]
        y = torch.tensor(y)

        if idx == self.__len__() - 1:
            self.shuffle_data()

        return X, y
    
    def shuffle_data(self):
        indices = np.random.permutation(self.__len__())
        self.Xs = self.Xs[indices]
        self.ys = self.ys[indices]
