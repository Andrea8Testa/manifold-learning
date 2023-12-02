import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import numpy as np


class MyDataset(Dataset):

    def __init__(self, file_name, s_index, e_index):

        # read csv file and load row data into variables
        file_out = pd.read_csv(file_name)

        n_rows = len(file_out)
        x = file_out.iloc[1:n_rows+1, s_index:e_index].values
        y = file_out.iloc[1:n_rows+1, s_index:e_index].values

        # Features Scaling
        sc = StandardScaler()
        x_train = sc.fit_transform(x)
        y_train = sc.fit_transform(y)
        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)

        # converting to torch tensors
        self.X_train = torch.tensor(x_train, dtype=torch.float32)
        self.y_train = torch.tensor(y_train, dtype=torch.float32)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]
