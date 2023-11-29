import torch
import numpy as np
from dataset import MyDataset
from vae_modules import VAE, RBF, train
import matplotlib.pyplot as plt

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
rbf = RBF(z_array, 2, 500)
rbf.load_parameters()

plt.scatter(z_array[:, 0], z_array[:, 1], alpha=0.7, c="black", s=0.05)

z1, z2 = np.meshgrid(np.linspace(min(z_array[:, 0]), max(z_array[:, 0]), 100),
                     np.linspace(min(z_array[:, 1]), max(z_array[:, 1]), 100))
z12 = np.column_stack([z1.flat, z2.flat])

variance = np.zeros([10000])
for it in range(0, 10000):
    var_inv = rbf.forward(z12[it])
    var = 10/(var_inv + 0.001)
    if var < 5:
        variance[it] = var
    else:
        variance[it] = 5

variance = variance.reshape(z1.shape)

plt.contourf(z1, z2, variance, cmap='viridis', alpha=0.3)

cbar = plt.colorbar()
cbar.set_label('Variance')

# Add labels and title
plt.xlabel('Z-axis 1')
plt.ylabel('Z-axis 2')
plt.title('Scatter Plot with Variance Background')
# Show the plot
plt.show()
