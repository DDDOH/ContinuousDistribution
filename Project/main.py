import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from model import model
from torch.autograd import Variable

import torch.distributions as D
from torch.distributions.multivariate_normal import MultivariateNormal


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import multivariate_normal
import pickle
from sklearn import preprocessing


from sklearn.model_selection import train_test_split
# y_data: conditon
# x_data: dependent

DATASET = 'stoxx' # 'stoxx' or '2d'
LAMBDA = 5000

if DATASET == 'stoxx':
    # read dataset
    import pandas as pd
    data = pd.read_csv('dataset/h_vstoxx.txt', header=2)
    data = data.dropna().values[:,1:].astype(float)
    total_len = data.shape[1]
    COND_LEN = 5
    DEP_LEN = total_len - COND_LEN
    n_observations = data.shape[0]
    X = data[:,:COND_LEN]
    Y = data[:,COND_LEN:]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=42)
    X_test, y_test = torch.tensor(X_test).float(), torch.tensor(y_test).float()
    
    

if DATASET == '2d':
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
    
    cond_dim = 1
    dependent_dim = 1

    fig, ax = plt.subplots()
    fig.set_size_inches(10, 8)
    sns.regplot(y_train, X_train, fit_reg=False)
    plt.savefig('toydata.png')
    plt.show()


training_set = np.concatenate([X_train, y_train], axis=1)
scaler = preprocessing.StandardScaler().fit(training_set)
training_set = scaler.transform(training_set)
training_set = torch.tensor(training_set).float()


# TODO: * update the network structure
# TODO: ** Continuous regularization

# Get model instance
network = model.MixtureDensityNetwork(dim_in=COND_LEN, dim_out=DEP_LEN, n_components=5)

# by default, the first serveral dimensions are the condition [X]
# the last parts are for the dependent term [Y].
# batch_size = 256


def normal_pdf(x, mean, var):
    normal = MultivariateNormal(loc=mean, covariance_matrix=torch.diag(var))
    return torch.exp(normal.log_prob(x))



def L2dist(pi_1, normal_1, pi_2, normal_2):
    n_component = pi_1.probs.shape[1]
    n_sample = pi_1.probs.shape[0]
    alpha = pi_1.probs
    beta = pi_2.probs
    mu = normal_1.loc
    eta = normal_2.loc
    Sigma = (normal_1.scale) ** 2
    Lambda = (normal_2.scale) ** 2
    
    regularization = torch.zeros(n_sample)
    for k in range(n_sample):
        k_th_loss = 0
        # A_n,n'
        for n in range(n_component):
            for n_prime in range(n_component):
                k_th_loss += alpha[k,n] * alpha[k,n_prime] * normal_pdf(x=mu[k,n,:], mean=mu[k,n_prime,:], var=Sigma[k,n,:] + Sigma[k,n_prime,:])
        # B_m,m'
        for m in range(n_component):
            for m_prime in range(n_component):
                k_th_loss += beta[k,m] * beta[k,m_prime] * normal_pdf(x=eta[k,m,:], mean=eta[k,m_prime,:], var=Lambda[k,m,:] + Lambda[k,m_prime,:])
        # C_n,m
        for n in range(n_component):
            for m in range(n_component):
                k_th_loss -= 2 * (alpha[k,n] * beta[k,m] * normal_pdf(x=mu[k,n,:], mean=eta[k,m], var=Sigma[k,n,:] + Lambda[k,m,:]))
        regularization[k] = k_th_loss
    return regularization


def L2regularization(inter, network):
    pi, normal = network(inter)
    n_sample = inter.shape[0]
    x_0 = inter
    x = x_0.clone().detach().requires_grad_(True)
    pi_1, normal_1 = network(x_0)
    pi_2, normal_2 = network(x)
    D_x_0 = torch.mean(L2dist(pi_1, normal_1, pi_2, normal_2))
    regularization_term = torch.norm(torch.autograd.grad(
        outputs=D_x_0, inputs=x, create_graph=True)[0])

    
    return regularization_term


def get_inter(x, n_points):
    n_cond = x.shape[0]
    x_1 = x[np.random.choice(n_cond, n_points, replace=False),:]
    x_2 = x[np.random.choice(n_cond, n_points, replace=False),:]
    alpha = torch.rand(n_points)
    alpha = torch.tile(alpha.unsqueeze(-1),(1,5))
    x = alpha * x_1 + (1 - alpha) * x_2
    return x    


def evaluate(network):
    test_loss = network.loss(X_test, y_test).mean()
    print('Test loss:{}'.format(test_loss))

def train(epochs, network):
    l2_points = 10
    
    # Train the model
    optimizer = torch.optim.Adam(network.parameters())
    for i in range(epochs):
        # for one training iteration
        optimizer.zero_grad()
        one_batch = training_set #[:batch_size, :]
        y = one_batch[:, -DEP_LEN:]
        x = one_batch[:, :COND_LEN]
        
        if i % 5 == 0:
            ori_loss = network.loss(x, y).mean()
            # inter is the abbreviation of interpolate
            inter = get_inter(x, n_points=l2_points)
            # randomly select several x locations
            regularization = torch.log(L2regularization(inter, network))
            loss = ori_loss + LAMBDA * regularization
            print('[%d/%d] Original Loss:%.4f\tRegularization:%.4f' % (i, epochs, ori_loss, regularization))
        else:
            ori_loss = network.loss(x, y).mean()
            loss = ori_loss
        loss.backward()
        print('[%d/%d] Original Loss:%.4f' % (i, epochs, loss))
        optimizer.step()

        if i % 10 == 0:
            evaluate(network)
    return network




network = train(epochs=100, network=network)
