import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
duration = 3  # seconds
frequency = 100  # Hz
num_points = duration * frequency
time_points = np.linspace(0, duration, num_points)

theta_xy = np.linspace(0, np.pi / 2, num_points).reshape(-1, 1)
radius_xy = 0.6
x = - radius_xy * np.cos(theta_xy)
y = -0.6 + radius_xy * np.sin(theta_xy)
z = 0.6 - radius_xy * (1 - np.cos(theta_xy))

x_4 = np.copy(x) - radius_xy/8 * np.sin(theta_xy*4)
y_4 = np.copy(y) + radius_xy/8 * np.sin(theta_xy*4)
z_4 = np.copy(z) - radius_xy/8 * (1-np.cos(theta_xy*4))

x_8 = np.copy(x) - radius_xy/16 * np.sin(theta_xy*8)
y_8 = np.copy(y) + radius_xy/16 * np.sin(theta_xy*8)
z_8 = np.copy(z) - radius_xy/16 * (1-np.cos(theta_xy*8))

# Combine trajectories
x_combined = np.vstack([x, x_4, x_8])
y_combined = np.vstack([y, y_4, y_8])
z_combined = np.vstack([z, z_4, z_8])

# Plot 3D trajectory
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_combined, y_combined, z_combined, label='3D Trajectory', s=0.2)
ax.scatter([x_combined[0]], [y_combined[0]], [z_combined[0]], color='red', label='Start Point')
ax.scatter([x_combined[-1]], [y_combined[-1]], [z_combined[-1]], color='green', label='End Point')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()

data = np.hstack([x_combined, y_combined, z_combined])

file_path = 'cartesian_motion.csv'
np.savetxt(file_path, data, delimiter=',', header='x,y,z', comments='')
