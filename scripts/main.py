import numpy as np
import torch
from vae_modules import VAE, RBF
from dijkistra_algorithm import Graph, dijkstra
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import CubicSpline, UnivariateSpline
from dataset import MyDataset

N = 50
contour_levels = 50
z1, z2 = np.meshgrid(np.linspace(-6, 6, N),
                     np.linspace(-6, 6, N))
z12 = np.column_stack([z1.flat, z2.flat])

model = torch.load("neural_networks/toy/toy_data_model_cuda")
z_array = np.load("neural_networks/toy/z_array.npy")
radial_bf = RBF(z_array, 2, 500, 2)
radial_bf.load_parameters("toy_parameters")

g = Graph(z12, model, radial_bf, 0.1)

start_point = np.array([0.5, 4])
start_index = np.where(np.sqrt(np.sum((z12-start_point)**2, axis=1)) < 0.2)[0][0]
end_point = np.array([-1, -2.5])
end_index = np.where(np.sqrt(np.sum((z12-end_point)**2, axis=1)) < 0.2)[0][0]
D = dijkstra(g, start_index, end_index)
print(D)
print(g.best_path[end_index])
planned_nodes = z12[g.best_path[end_index]]

# Create a CubicSpline object for interpolation
cs_z1 = UnivariateSpline(np.linspace(0, 1, planned_nodes.shape[0]), planned_nodes[:, 0], s=0.3)
cs_z2 = UnivariateSpline(np.linspace(0, 1, planned_nodes.shape[0]), planned_nodes[:, 1], s=0.3)
planned_trajectory_z1 = cs_z1(np.linspace(0, 1, 10*planned_nodes.shape[0])).reshape(-1, 1)
planned_trajectory_z2 = cs_z2(np.linspace(0, 1, 10*planned_nodes.shape[0])).reshape(-1, 1)
planned_trajectory = np.hstack([planned_trajectory_z1, planned_trajectory_z2])

M = np.zeros([int(N*N)])
for it in range(0, int(N*N)):
    JsTJs = np.matmul(radial_bf.gradient(z12[it]), radial_bf.gradient(z12[it]).T)
    z_tensor = torch.tensor(z12[it], dtype=torch.float32, requires_grad=True).to(model.device)
    Jm = torch.zeros([3, 2])
    Jm[0, :] = torch.autograd.grad(model.decode(z_tensor)[0], z_tensor, retain_graph=True)[0]
    Jm[1, :] = torch.autograd.grad(model.decode(z_tensor)[1], z_tensor, retain_graph=True)[0]
    Jm[2, :] = torch.autograd.grad(model.decode(z_tensor)[2], z_tensor, retain_graph=True)[0]
    JmTJm = torch.matmul(Jm.T, Jm).numpy()
    M_matrix = JsTJs + JmTJm
    M[it] = np.sqrt(np.linalg.det(M_matrix))
M = M.reshape(z1.shape)

plt.figure()
plt.scatter(z_array[:, 0], z_array[:, 1], alpha=0.9, c="black", s=0.05)
plt.plot(planned_trajectory[:, 0], planned_trajectory[:, 1], alpha=0.9, c="blue")
plt.contourf(z1, z2, M, contour_levels, cmap='magma', alpha=0.3)
cbar = plt.colorbar()
cbar.set_label('Magnification Factor')
# Add labels and title
plt.xlabel('Z-axis 1')
plt.ylabel('Z-axis 2')
plt.show()

Cartesian_traj = np.zeros([planned_trajectory.shape[0], 3])
for it in range(planned_trajectory.shape[0]):
    z_tensor = torch.tensor(planned_trajectory[it, :], dtype=torch.float32, requires_grad=True).to(model.device)
    Cartesian_traj[it, :] = model.decode(z_tensor).cpu().detach().numpy()

data = MyDataset('data/toy_data.csv', 0, 3)
Cartesian_traj_n = Cartesian_traj*data.std + data.mean

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Cartesian_traj_n[:, 0], Cartesian_traj_n[:, 1], Cartesian_traj_n[:, 2], label='3D Trajectory', s=0.2)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()
