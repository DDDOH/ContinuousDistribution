import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

import torch.distributions as D
from torch.distributions.multivariate_normal import MultivariateNormal

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import multivariate_normal
import pickle


from sklearn.model_selection import train_test_split
# y_data: conditon
# x_data: dependent

# get synthetic dataset


def build_toy_dataset(n):
    np.random.seed(41)
    y_data = np.random.uniform(-10.5, 10.5, n)
    r_data = np.random.normal(size=n)  # random noise
#     x_data = np.sin(0.75 * y_data) * 7.0 + y_data * 0.5 + r_data * 1.0
    x_data = 0.2 * y_data ** 2 + r_data * 10 / (abs(y_data) + 5)
    x_data = x_data.reshape((n, 1))
    y_data = y_data.reshape((n, 1))

    return train_test_split(x_data, y_data, random_state=42)


n_observations = 5000  # number of data points

X_train, X_test, y_train, y_test = build_toy_dataset(n_observations)

fig, ax = plt.subplots()
fig.set_size_inches(10, 8)
sns.regplot(y_train, X_train, fit_reg=False)
plt.savefig('toydata.png')
plt.show()


training_set = torch.tensor(np.concatenate([X_train, y_train], axis=1))

# TODO: * update the network structure
# TODO: ** Continuous regularization


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
        self.z_sigma = nn.Linear(
            n_hidden, n_gaussians * dependent_dim)
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
        sigma_diagnoal = torch.exp(self.z_sigma(z_h)).reshape(
            (batch_size, self.n_gaussians, self.dependent_dim))
        sigma = torch.zeros((batch_size, self.n_gaussians,
                            self.dependent_dim, self.dependent_dim))
        # TODO more efficient implementation
        for i in range(batch_size):
            for j in range(self.n_gaussians):
                sigma[i, j, :, :] = torch.diag(sigma_diagnoal[i, j, :])
        mu = self.z_mu(z_h).reshape(
            (batch_size, self.n_gaussians, self.dependent_dim))

        return pi, sigma, mu

# utils


def loss_function(pi, sigma, mu, y):
    log_prob = 0
    # TODO more efficient implementation
    for i in range(batch_size):
        pi_i = pi[i, :]
        sigma_i = sigma[i, :, :, :]
        mu_i = mu[i, :, :]
        mix = D.Categorical(pi_i)
        comp = D.Independent(D.MultivariateNormal(
            loc=mu_i, covariance_matrix=sigma_i), reinterpreted_batch_ndims=0)
        gmm_torch = D.MixtureSameFamily(mix, comp)
        log_prob += gmm_torch.log_prob(y[i, :].float())
    return - log_prob


def evaluate(x):
    # TODO: *** compare the real conditional distribution with the learned conditional distribution
    pi, sigma, mu = network(x.float())

    # CAN USE TEST LOGLIKELIHOOD OR COMPUTE THE W-DISTANCE

    expected_dist = 0
    n_sample = np.shape(x)[0]
    for i in range(n_sample):
        real_cond_gmm = gmm.get_cond_gm(x[i, :])
        network_cond_gmm = GaussianMixture(
            weight_ls=pi[i, :], mean_ls=mu[i, :, :], cov_ls=sigma[i, :, :, :])
        expected_dist += gmm_dist(real_cond_gmm, network_cond_gmm)/n_sample
    return expected_dist


def gmm_dist(gmm_1, gmm_2):
    """Compute the w-distance between two gaussian mixture model.

    Args:
        gmm_1 (GaussianMixture): The first GMM.
        gmm_2 (GaussianMixture): The second GMM.
    """
    # TODO: distance between two gmm.
    raise NotImplementedError


# Get model instance
cond_dim = 1
dependent_dim = 1
network = MDN(cond_dim=cond_dim, dependent_dim=dependent_dim,
              n_hidden=20, n_gaussians=5)


# by default, the first serveral dimensions are the condition [X]
# the last parts are for the dependent term [Y].
batch_size = 256


def train(epochs, network):
    # Train the model
    optimizer = torch.optim.Adam(network.parameters())
    for i in range(epochs):
        # for one training iteration
        optimizer.zero_grad()
        # TODO modify how we get one batch of data
        one_batch = training_set[:batch_size, :]
        y = one_batch[:, -dependent_dim:]
        x = one_batch[:, :cond_dim]
        pi, sigma, mu = network(x.float())
        loss = loss_function(pi, sigma, mu, y)
        loss.backward()
        print('[%d/%d] Loss:%.4f' % (i, epochs, loss))
        optimizer.step()

        if i % 10 == 0:
            plot()
        # if i % 10 == 0:
        #     evaluate(x)
    return network


def plot():


network = train(epochs=100, network=network)


# mix = D.Categorical(torch.rand(3,5))
# comp = D.Independent(D.Normal(
#             torch.randn(3,5,2), torch.rand(3,5,2)), 1)
# gmm_torch = D.MixtureSameFamily(mix, comp)

# loss = mdn_loss_fn(pi, sigma, mu, y)
# TODO use gmm_torch.log_prob() as loss function


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


# def gaussian_distribution(y, mu, sigma):
#     result = (y.expand_as(mu) - mu) * torch.reciprocal(sigma)
#     result = -0.5*(result * result)
#     return torch.exp(result)*torch.reciprocal(sigma)


# def mdn_loss_fn(pi, sigma, mu, y):
#     result = gaussian_distribution(y, mu, sigma)*pi
#     result = torch.sum(result, dim=1)
#     result = -torch.log(result)
#     return torch.mean(result)
