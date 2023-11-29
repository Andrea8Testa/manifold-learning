import torch
from dataset import MyDataset
from vae_modules import VAE, train
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print("device: ", device)
epochs = 50
batch_size = 100

data = MyDataset('x.csv')
train_loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=True)

model = VAE(device).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
train(model, optimizer, epochs, batch_size, train_loader, device)


