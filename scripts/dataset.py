import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class MyDataset(Dataset):

    def __init__(self, file_name):

        # read csv file and load row data into variables
        file_out = pd.read_csv(file_name)

        n_rows = len(file_out)
        x = file_out.iloc[1:n_rows+1, 1:4].values
        y = file_out.iloc[1:n_rows+1, 1:4].values

        # Features Scaling
        sc = StandardScaler()
        x_train = sc.fit_transform(x)
        y_train = sc.fit_transform(y)

        # converting to torch tensors
        self.X_train = torch.tensor(x_train, dtype=torch.float32)
        self.y_train = torch.tensor(y_train, dtype=torch.float32)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]


