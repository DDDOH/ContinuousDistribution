import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

import torch.distributions as D
from torch.distributions.multivariate_normal import MultivariateNormal


class GaussianMixture:
    def __init__(self, weight_ls, mean_ls, cov_ls):
        """[summary]

        Args:
            weight_ls (np.array or list): 1D array of length n_components.
            mean_ls (np.array): 2D array of size (n_components, dim)
            cov_ls (list): list of covariance matrix.
        """
        assert abs(np.sum(weight_ls) - 1) < 1e-5 and np.min(weight_ls) > 0
        self.n_components = len(weight_ls)
        self.dim = np.shape(mean_ls)[1]
        self.weight_ls = weight_ls
        self.mean_ls = mean_ls
        self.cov_ls = cov_ls

    def sample(self, n_sample):
        sample_ls = np.zeros((n_sample, self.dim))
        for i in range(n_sample):
            which_component = np.random.choice(
                self.n_components, p=self.weight_ls)
            one_training_sample = np.random.multivariate_normal(
                mean=self.mean_ls[which_component, :], cov=self.cov_ls[which_component])
            sample_ls[i, :] = one_training_sample
        self.n_sample = n_sample
        self.sample_ls = sample_ls
        return sample_ls


n_components = 5
dim = 10
weight_ls = np.array([0.2, 0.1, 0.3, 0.3, 0.1])
mean_ls = np.random.uniform(20, 30, size=(n_components, dim))

cov_ls = []
for i in range(n_components):
    # get a cov matrix
    _ = np.random.uniform(-1, 1, size=(dim, dim))
    i_cov = np.matmul(_, _.T)
    cov_ls.append(i_cov)
model = GaussianMixture(weight_ls=weight_ls, mean_ls=mean_ls, cov_ls=cov_ls)


n_sample = 300
training_set = torch.tensor(model.sample(
    n_sample=n_sample), dtype=torch.float64)

mix = D.Categorical(torch.ones(n_components,))
comp = D.Independent(D.MultivariateNormal(loc=torch.tensor(
    mean_ls), covariance_matrix=torch.tensor(np.stack(cov_ls))), reinterpreted_batch_ndims=0)
gmm = D.MixtureSameFamily(mix, comp)

gmm.sample(sample_shape=torch.Size([100]))


class MDN(nn.Module):
    def __init__(self, cond_dim, dependent_dim, n_hidden, n_gaussians):
        """[summary]

        Args:
            cond_dim (int): The dimension of condition.
            dependent_dim (int): The dimension of dependent term.
            n_hidden ([type]): [description]
            n_gaussians ([type]): The number of components for the GMM.
        """
        super(MDN, self).__init__()
        self.cond_dim = cond_dim
        self.dependent_dim = dependent_dim
        self.n_gaussians = n_gaussians
        self.z_h = nn.Sequential(
            nn.Linear(cond_dim, n_hidden),
            nn.Tanh()
        )
        self.z_pi = nn.Linear(n_hidden, n_gaussians)
        # if assume the covariance matrix is diagnoal matrix
        # z_sigma should have shape (n_gaussians, dependent_dim)
        self.z_sigma = nn.Linear(n_hidden, n_gaussians * dependent_dim)
        # z_mu should have shape (n_gaussians, dependent_dim)
        self.z_mu = nn.Linear(n_hidden, n_gaussians * dependent_dim)

    def forward(self, x):
        """Calculate weight, mean and covariance matrix of the GMM.

        Args:
            x ([type]): [description]

        Returns:
            [type]: [description]
        """
        if x.ndim == 1:
            x = x.unsqueeze(0)
        batch_size = x.shape[0]
        z_h = self.z_h(x)
        pi = nn.functional.softmax(self.z_pi(z_h), -1)
        sigma = torch.exp(self.z_sigma(z_h)).reshape(
            (batch_size, self.n_gaussians, self.dependent_dim))
        mu = self.z_mu(z_h).reshape(
            (batch_size, self.n_gaussians, self.dependent_dim))
        return pi, sigma, mu


def gaussian_distribution(y, mu, sigma):
    result = (y.expand_as(mu) - mu) * torch.reciprocal(sigma)
    result = -0.5*(result * result)
    return torch.exp(result)*torch.reciprocal(sigma)


def mdn_loss_fn(pi, sigma, mu, y):
    result = gaussian_distribution(y, mu, sigma)*pi
    result = torch.sum(result, dim=1)
    result = -torch.log(result)
    return torch.mean(result)


cond_dim = 4
dependent_dim = 6
network = MDN(cond_dim=cond_dim, dependent_dim=dependent_dim,
              n_hidden=20, n_gaussians=5)
optimizer = torch.optim.Adam(network.parameters())
# set the data

# by default, the first serveral dimensions are the condition [X]
# the last parts are for the dependent term [Y].

# for one training iteration
batch_size = 256
one_batch = training_set[:batch_size, :]
y = one_batch[:, -dependent_dim:]
x = one_batch[:, :cond_dim]
pi, sigma, mu = network(x.float())
loss = mdn_loss_fn(pi, sigma, mu, y)
# TODO use gmm.log_prob() as loss function


# optimizer.zero_grad()
# loss.backward()
# optimizer.step()


# def train_mdn():
#     for epoch in range(2000):
#         # 经过神经网络
#         loss = mdn_loss_fn()
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()


# train_mdn()
