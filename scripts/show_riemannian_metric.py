import torch
import numpy as np
from vae_modules import VAE, RBF, train
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)

# import trained vae
if device == torch.device("cuda"):
    model = torch.load("neural_networks/toy/toy_data_model_cuda")
else:
    model = torch.load("neural_networks/toy/trained_model_cpu")

# import latent data
z_array = np.load("neural_networks/toy/z_array.npy")

# import trained rbf network
rbf = RBF(z_array, 2, 500, 2)
rbf.load_parameters("toy_parameters")

N = 100
contour_levels = 50
z1, z2 = np.meshgrid(np.linspace(min(z_array[:, 0]), max(z_array[:, 0]), N),
                     np.linspace(min(z_array[:, 1]), max(z_array[:, 1]), N))
z12 = np.column_stack([z1.flat, z2.flat])


plt.figure()

M = np.zeros([int(N*N)])
for it in range(0, int(N*N)):
    JsTJs = np.matmul(rbf.gradient(z12[it]), rbf.gradient(z12[it]).T)
    z_tensor = torch.tensor(z12[it], dtype=torch.float32, requires_grad=True).to(device)
    Jm = torch.zeros([3, 2])
    Jm[0, :] = torch.autograd.grad(model.decode(z_tensor)[0], z_tensor, retain_graph=True)[0]
    Jm[1, :] = torch.autograd.grad(model.decode(z_tensor)[1], z_tensor, retain_graph=True)[0]
    Jm[2, :] = torch.autograd.grad(model.decode(z_tensor)[2], z_tensor, retain_graph=True)[0]
    JmTJm = torch.matmul(Jm.T, Jm).numpy()
    M_matrix = JsTJs + JmTJm
    M[it] = np.sqrt(np.linalg.det(M_matrix))
M = M.reshape(z1.shape)

plt.scatter(z_array[:, 0], z_array[:, 1], alpha=0.9, c="black", s=0.05)
plt.contourf(z1, z2, M, contour_levels, cmap='magma', alpha=0.3)
cbar = plt.colorbar()
cbar.set_label('Magnification Factor')
# Add labels and title
plt.xlabel('Z-axis 1')
plt.ylabel('Z-axis 2')
plt.show()
