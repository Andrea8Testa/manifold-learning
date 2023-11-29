import torch
import numpy as np
from dataset import MyDataset
from vae_modules import VAE, RBF, train
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)
# import data
data = MyDataset('x.csv')
# import trained vae
model = torch.load("trained_model")
# generate latent dataset
"""z_array = np.zeros([data.__len__(), 2])
for it in range(0, data.__len__()):
    _, latent_distribution = model.forward(data.X_train[it].to(device))
    latent_mean = latent_distribution.loc.detach().cpu().numpy()
    z_array[it, :] = latent_mean"""
# import latent data
z_array = np.load("z_array.npy")
variance = RBF(z_array, 2, 500)
