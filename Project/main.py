import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

import torch.distributions as D
from torch.distributions.multivariate_normal import MultivariateNormal

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
import pickle


class GaussianMixture():
    def __init__(self, weight_ls, mean_ls, cov_ls):
        # cov_ls is a list of numpy array, each array has size n * n
        # n is the number of components
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

    def get_cond_gm(self, x_known):
        c_dim = len(x_known)

        def get_conditional_mean_cov_multivariate_gaussian(x_known, mean, cov):
            # calculate conditional distribution for multivariate gaussian distribution
            # default knowning the first C_DIM dimensions
            # x_known: the known value at the first C_DIM dimensions
            # mean: a list, the mean for each dimension of the multivariate gaussian distribution
            # cov: a matrix, the covariance matrix of the multivariate gaussian distribution
            # know the first c_dim dimensions
            mean_k = mean[:c_dim]
            mean_u = mean[c_dim:]
            cov_kk = cov[:c_dim, :c_dim]
            cov_ku = cov[:c_dim, c_dim:]
            cov_uk = cov[c_dim:, :c_dim]
            cov_uu = cov[c_dim:, c_dim:]

            mean_cond = mean_u + \
                cov_uk.dot(np.linalg.inv(cov_kk)).dot((x_known - mean_k).T).T
            cov_cond = cov_uu - cov_uk.dot(np.linalg.inv(cov_kk)).dot(cov_ku)
            return mean_cond, cov_cond

        marginal = 0
        for l in range(self.n_components):
            mean_l = self.mean_ls[l, :]
            cov_l = self.cov_ls[l]

            mean_l_known = mean_l[:c_dim]
            cov_l_known = cov_l[:c_dim, :c_dim]
            marginal += self.weight_ls[l] * multivariate_normal.pdf(
                x_known, mean=mean_l_known, cov=cov_l_known)

        # cov, mean and weight for each components
        cond_pi_ls = np.zeros(self.n_components)
        cond_mean_ls = np.zeros((self.n_components, self.dim - c_dim))
        cond_cov_ls = []
        # the k-th component of the conditional gaussian mixture distribution
        for k in range(self.n_components):
            mean_k = self.mean_ls[k, :]
            cov_k = self.cov_ls[k]

            # conditional mean and covariance matrix for k-th component
            cond_mean_k, cond_cov_k = get_conditional_mean_cov_multivariate_gaussian(
                x_known, mean=mean_k, cov=cov_k)
            cond_mean_ls[k, :] = cond_mean_k
            cond_cov_ls.append(cond_cov_k)

            mean_k_known = mean_k[:c_dim]
            cov_k_known = cov_k[:c_dim, :c_dim]

            # weight for k-th component for the conditional GMM
            cond_pi_ls[k] = self.weight_ls[k] * multivariate_normal.pdf(
                x_known, mean=mean_k_known, cov=cov_k_known) / marginal

        cond_gm = GaussianMixture(
            weight_ls=cond_pi_ls, mean_ls=cond_mean_ls, cov_ls=cond_cov_ls)
        return cond_gm

    def save_model(self, path):
        para_name = '_DIM_{}_N_GAUSSIAN_{}_TRAINING_SIZE_{}.pkl'.format(
            self.dim, self.n_components, self.n_sample)
        # Overwrites any existing file.
        with open(path + para_name, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
        return path + para_name


# TODO: * update the network structure
# TODO: ** Continuous regularization
# Define synthetic GMM
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

gmm = GaussianMixture(weight_ls, mean_ls, cov_ls)

# mix = D.Categorical(torch.ones(n_components,))
# comp = D.Independent(D.MultivariateNormal(loc=torch.tensor(
#     mean_ls), covariance_matrix=torch.tensor(np.stack(cov_ls))), reinterpreted_batch_ndims=0)
# gmm_torch = D.MixtureSameFamily(mix, comp)

mix = D.Categorical(torch.tensor(weight_ls))
comp = D.Independent(D.MultivariateNormal(loc=torch.tensor(
    mean_ls), covariance_matrix=torch.tensor(np.stack(cov_ls))), reinterpreted_batch_ndims=0)
gmm_torch = D.MixtureSameFamily(mix, comp)

n_sample = 300

# get training set
training_set = gmm_torch.sample(sample_shape=torch.Size([n_sample]))
# gmm_torch.log_prob(training_set)


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
        if i % 10 == 0:
            evaluate(x)
    return - log_prob


def evaluate(x):
    # TODO: *** compare the real conditional distribution with the learned conditional distribution
    pi, sigma, mu = network(x.float())

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
cond_dim = 4
dependent_dim = 6
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
    return network


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
