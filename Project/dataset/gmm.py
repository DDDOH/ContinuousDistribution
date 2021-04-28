
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
