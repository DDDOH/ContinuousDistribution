"""Test gradient penalty works on 1D function prediction.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import progressbar
import matplotlib.pyplot as plt


np.random.seed(0)

if not os.path.exists('result'):
    os.makedirs('result')

# The true function
def func(x):
    return np.sin(x).astype(np.float32)

# Create training set using the true function. Noise is added to the true function value.
def get_training_set(size=20):
    x = np.linspace(0,10,size).astype(np.float32)
    y = func(x) + np.random.normal(0,0.2,size=size)
    y = y.astype(np.float32)
    x, y = torch.tensor(x).unsqueeze(-1), torch.tensor(y).unsqueeze(-1)
    return x, y


x, y = get_training_set(20)

loss_func = torch.nn.MSELoss()

# Use a three layers MLP to fit the training set.
class NET(nn.Module):
    def __init__(self, input_size):
        super(NET, self).__init__()
        self.net_size = 128
        self.linear = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=self.net_size),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(in_features=self.net_size, out_features=self.net_size),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(in_features=self.net_size, out_features=1)
        )

    def forward(self, x):
        return self.linear(x)
    
pred = NET(input_size=1)
optimizer = torch.optim.Adam(pred.parameters(), lr=0.005)

# The function to compute gradient penalty. Note that create_graph must be set to True so that the computed gradient can be 'backwarded' and thus being minimized.
def gradient_penalty():
    n_loc = 100
    x_loc = (torch.rand(n_loc) * 10).unsqueeze(-1).requires_grad_(True)
    pred_x_loc = pred(x_loc)
    weight = torch.ones(pred_x_loc.size())
    grad_net_x = torch.autograd.grad(outputs=pred_x_loc, inputs=x_loc, grad_outputs=weight, create_graph=True)
    return x_loc.squeeze(-1).detach(), grad_net_x[0].squeeze(-1), torch.sum(grad_net_x[0]**2).mean()

EPOCHS = 10000 # training epochs
LAMBDA = 0 # or 0.01 seems properly # the weight of the gradient penalty term

# the training iteration
for i in progressbar.progressbar(range(EPOCHS), redirect_stdout=True):
    pred_y = pred(x)
    x_loc, grad_net_x, grad_penalty_val = gradient_penalty()
    pred_loss = loss_func(y, pred_y)
    scaled_grad_penalty = grad_penalty_val * LAMBDA
    loss = pred_loss + scaled_grad_penalty
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('Total: {}\tPred Loss:{}\tGrad Loss:{}'.format(loss, pred_loss, grad_penalty_val))
    if i % 500 == 0:
        plt.figure()
        plt.subplot(211)
        x_line = torch.tensor(np.arange(0,10,10/1000).astype(np.float32)).unsqueeze(-1)
        pred_y_line = pred(x_line)
        plt.plot(x_line.squeeze(-1), pred_y_line.detach().squeeze(-1),label='Predicted')
        plt.scatter(x, y,label='True')
        plt.legend()
        
        plt.subplot(212)
        plt.scatter(x_loc, grad_net_x.detach(), s=5)
        plt.hlines(y=0, xmin=0, xmax=10)
        plt.savefig('result/pred_{}.jpg'.format(i))
    
