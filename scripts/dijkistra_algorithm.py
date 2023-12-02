from queue import PriorityQueue
import numpy as np
import torch
from vae_modules import VAE, RBF


class Graph:
    def __init__(self, z_grid, nn: VAE, rbf: RBF, d_th=0.02):
        self.v = z_grid.shape[0]
        self.d_th = d_th
        # self.edges = [[-1 for i in range(num_of_vertices)] for j in range(num_of_vertices)]
        self.z_grid = z_grid
        self.visited = []
        self.best_path = [[] for i in range(self.v)]

        # import trained networks
        self.nn = nn
        self.rbf = rbf

    # def add_edge(self, u, v, weight):
    #    self.edges[u][v] = weight
    #    self.edges[v][u] = weight


def dijkstra(graph, start_vertex, end_vertex):
    d = {v: float('inf') for v in range(graph.v)}
    d[start_vertex] = 0

    pq = PriorityQueue()
    pq.put((0, start_vertex))
    graph.best_path = [[start_vertex] for i in range(graph.v)]

    while not pq.empty():
        (dist, current_vertex) = pq.get()
        graph.visited.append(current_vertex)
        if current_vertex == end_vertex:
            return d[end_vertex]

        neighborhood = np.where(np.sqrt(np.sum((graph.z_grid[current_vertex]-graph.z_grid)**2, axis=1) < graph.d_th))[0]
        neighborhood_ = np.delete(neighborhood, np.where(neighborhood == current_vertex))
        if neighborhood_.size > 4:
            print("warning: d_th is too large")
        elif neighborhood_.size < 2:
            print("warning: d_th is too small")
        for neighbor in neighborhood_:
            z = (graph.z_grid[current_vertex] + graph.z_grid[neighbor])/2
            delta_z = graph.z_grid[neighbor] - graph.z_grid[current_vertex]
            JsTJs = np.matmul(graph.rbf.gradient(z), graph.rbf.gradient(z).T)
            z_tensor = torch.tensor(z, dtype=torch.float32, requires_grad=True).to(graph.nn.device)
            Jm = torch.zeros([3, 2])
            Jm[0, :] = torch.autograd.grad(graph.nn.decode(z_tensor)[0], z_tensor, retain_graph=True)[0]
            Jm[1, :] = torch.autograd.grad(graph.nn.decode(z_tensor)[1], z_tensor, retain_graph=True)[0]
            Jm[2, :] = torch.autograd.grad(graph.nn.decode(z_tensor)[2], z_tensor, retain_graph=True)[0]
            JmTJm = torch.matmul(Jm.T, Jm).numpy()
            M_matrix = JsTJs + JmTJm
            distance = np.sqrt(np.matmul(np.matmul(delta_z.T, M_matrix), delta_z))
            if neighbor not in graph.visited:
                old_cost = d[neighbor]
                new_cost = d[current_vertex] + distance
                if new_cost < old_cost:
                    pq.put((new_cost, neighbor))
                    d[neighbor] = new_cost
                    graph.best_path[neighbor] = graph.best_path[current_vertex].copy()
                    graph.best_path[neighbor].append(neighbor)
    return d


"""N = 10
z1, z2 = np.meshgrid(np.linspace(-6, 6, N),
                     np.linspace(-6, 6, N))
z12 = np.column_stack([z1.flat, z2.flat])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)

model = torch.load("neural_networks/toy/toy_data_model_cuda")
z_array = np.load("neural_networks/toy/z_array.npy")
radial_bf = RBF(z_array, 2, 500, 2)
radial_bf.load_parameters("toy_parameters")

g = Graph(z12, model, radial_bf, 2)
start = 0
end = 80
D = dijkstra(g, start, end)
print(D)
print(g.best_path[end])"""

"""g = Graph(9)
g.add_edge(0, 1, 4)
g.add_edge(0, 6, 7)
g.add_edge(1, 6, 11)
g.add_edge(1, 7, 20)
g.add_edge(1, 2, 9)
g.add_edge(2, 3, 6)
g.add_edge(2, 4, 2)
g.add_edge(3, 4, 10)
g.add_edge(3, 5, 5)
g.add_edge(4, 5, 15)
g.add_edge(4, 7, 1)
g.add_edge(4, 8, 5)
g.add_edge(5, 8, 12)
g.add_edge(6, 7, 1)
g.add_edge(7, 8, 3)

start = 0
end = 2

D = dijkstra(g, start, end)

print(D)
print(g.best_path[end])"""
