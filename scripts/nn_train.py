import torch
from dataset import MyDataset
from vae_modules import VAE, train
from torch.utils.data import DataLoader
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print("device: ", device)
epochs = 100
batch_size = 100

data = MyDataset('data/toy_data.csv', 0, 3)
train_loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=True)

model = VAE(device).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
train(model, optimizer, epochs, batch_size, train_loader, device)

data_0 = data.X_train[0].clone().to(device).to(dtype=torch.float32)
mean_0, _ = model.forward(data_0)
error_0 = mean_0 - data_0

data_400 = data.X_train[400].clone().to(device).to(dtype=torch.float32)
mean_400, _ = model.forward(data_400)
error_400 = mean_400 - data_400

data_800 = data.X_train[800].clone().to(device).to(dtype=torch.float32)
mean_800, _ = model.forward(data_800)
error_800 = mean_800 - data_800

z_array = np.zeros([data.__len__(), 2])
for it in range(0, data.__len__()):
    _, latent_distribution = model.forward(data.X_train[it].to(device))
    latent_mean = latent_distribution.loc.detach().cpu().numpy()
    z_array[it, :] = latent_mean

# torch.save(model, "neural_networks/toy/toy_data_model_cuda")
# np.save("neural_networks/toy/z_array", z_array)
