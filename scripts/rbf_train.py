import torch
import numpy as np
from dataset import MyDataset
from vae_modules import VAE, RBF, train
import matplotlib.pyplot as plt

z_array = np.load("neural_networks/toy/z_array.npy")
rbf = RBF(z_array, 2, 500, 2)
rbf.fitting()

N = 100
contour_levels = 50
z1, z2 = np.meshgrid(np.linspace(-6, 6, N),
                     np.linspace(-6, 6, N))
z12 = np.column_stack([z1.flat, z2.flat])

variance_inv = np.zeros([int(N*N)])
JsTJs = np.zeros([int(N*N)])
for it in range(0, int(N*N)):
    variance_inv[it] = rbf.forward(z12[it])
    JsTJs[it] = np.matmul(rbf.gradient(z12[it]), rbf.gradient(z12[it]).T)

variance_inv = variance_inv.reshape(z1.shape)
JsTJs = JsTJs.reshape(z1.shape)

plt.figure()
plt.scatter(z_array[:, 0], z_array[:, 1], alpha=0.9, c="black", s=0.05)
plt.contourf(z1, z2, variance_inv, contour_levels, cmap='magma', alpha=0.3)
cbar = plt.colorbar()
cbar.set_label('Variance')
# Add labels and title
plt.xlabel('Z-axis 1')
plt.ylabel('Z-axis 2')
plt.title('Scatter Plot with Variance Background')
# Show the first plot
plt.show()

plt.figure()
plt.scatter(z_array[:, 0], z_array[:, 1], alpha=0.9, c="black", s=0.05)
plt.contourf(z1, z2, JsTJs, contour_levels, cmap='magma', alpha=0.3)
cbar = plt.colorbar()
cbar.set_label('Gradient')
# Add labels and title
plt.xlabel('Z-axis 1')
plt.ylabel('Z-axis 2')
plt.title('Scatter Plot with Gradient Background')
# Show the first plot
plt.show()

# rbf.save_parameters("toy_parameters")

