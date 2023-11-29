import torch
import numpy as np
import torch.nn as nn
from torch.distributions import Normal, kl
from sklearn.cluster import KMeans
from numpy.linalg import inv


class VAE(nn.Module):

    def __init__(self, device, input_dim=3, f_hidden_dim=200, s_hidden_dim=100, latent_dim=2):
        super(VAE, self).__init__()

        self.device = device
        self.prior = Normal(torch.zeros(latent_dim).to(self.device),
                            torch.ones(latent_dim).to(self.device))

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, f_hidden_dim),
            nn.Softplus(),
            nn.Linear(f_hidden_dim, s_hidden_dim),
            nn.Softplus()
            )

        # latent mean and variance
        self.latent_mean_layer = nn.Linear(s_hidden_dim, latent_dim)
        self.latent_logvar_layer = nn.Linear(s_hidden_dim, latent_dim)

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, s_hidden_dim),
            nn.Softplus(),
            nn.Linear(s_hidden_dim, f_hidden_dim),
            nn.Softplus()
            )

        # mean and variance
        self.mean_layer = nn.Linear(f_hidden_dim, input_dim)
        # self.logvar_layer = nn.Linear(f_hidden_dim, input_dim)

    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = self.latent_mean_layer(x), self.latent_logvar_layer(x)
        return mean, logvar

    def decode(self, x):
        x = self.decoder(x)
        mean = self.mean_layer(x)
        return mean

    def forward(self, x):
        latent_mean, latent_log_var = self.encode(x)
        z_given_x = Normal(loc=latent_mean, scale=latent_log_var.exp())
        sample_z = z_given_x.rsample()
        mean = self.decode(sample_z)
        return mean, z_given_x

    def loss_function(self, x):
        mean, z_given_x = self.forward(x)
        kl_div = kl.kl_divergence(z_given_x, self.prior)
        x_given_z = Normal(loc=mean, scale=1e-3*torch.ones(mean.size()).to(self.device))
        rec_loss = torch.sum(x_given_z.log_prob(x), dim=1, keepdim=True)
        return -(rec_loss - kl_div).squeeze(dim=1)


class RBF:

    def __init__(self, z_array, latent_dim=2, kernels=500):
        # radial basis function
        self.z_array = z_array
        self.n_data = self.z_array.shape[0]
        self.n_clusters = kernels
        self.centers = np.zeros([self.n_clusters, latent_dim])
        self.weights = np.zeros(self.n_clusters)
        self.m_gram = np.zeros([self.n_data, self.n_clusters])
        self.gamma = 1

        self.parameters = {"centers": self.centers, "weights": self.weights, "m_gram": self.m_gram}

    def fitting(self):
        self.kmeans()
        self.compute_gram_matrix()
        self.train_weights()

    def kmeans(self):
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0, n_init="auto").fit(self.z_array)
        self.centers = kmeans.cluster_centers_

    def compute_gram_matrix(self):
        for it_data in range(0, self.n_data):
            for it_clusters in range(0, self.n_clusters):
                distance = np.sqrt(sum((self.z_array[it_data, :] - self.centers[it_clusters, :])**2))
                expo = np.exp(-self.gamma*distance)
                self.m_gram[it_data, it_clusters] = expo

    def train_weights(self):
        y = 10*np.ones(self.n_data)
        pseudo_inv = np.matmul(inv(np.matmul(self.m_gram.T, self.m_gram)), self.m_gram.T)
        self.weights = np.matmul(pseudo_inv, y)

    def forward(self, z):
        exp_vec = np.zeros(self.n_clusters)
        for it_clusters in range(0, self.n_clusters):
            distance = np.sqrt(sum((z - self.centers[it_clusters, :])**2))
            expo = np.exp(-self.gamma*distance)
            exp_vec[it_clusters] = expo
        y = sum(self.weights * exp_vec)
        return y

    def save_parameters(self):
        np.save("rbf_parameters", self.parameters)

    def load_parameters(self):
        self.parameters = np.load("rbf_parameters")
        self.centers = self.parameters["centers"]
        self.weights = self.parameters["weights"]
        self.m_gram = self.parameters["m_gram"]


def train(model, optimizer, epochs, batch_size, train_loader, device, x_dim=3):
    model.train()
    for epoch in range(epochs):
        overall_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            try:
                x = x.view(batch_size, x_dim).to(device)

                optimizer.zero_grad()
                loss = model.loss_function(x).mean()

                overall_loss += loss.item()

                loss.backward()
                optimizer.step()
            except:
                print("")

        print("\tEpoch", epoch + 1, "\tAverage Loss: ", overall_loss/(batch_idx*batch_size))
    return overall_loss
