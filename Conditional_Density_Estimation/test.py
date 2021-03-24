from cde.density_simulation import SkewNormal
from cde.density_estimator import KernelMixtureNetwork
import numpy as np

""" simulate some data """
density_simulator = SkewNormal(random_seed=22)
X, Y = density_simulator.simulate(n_samples=3000)

""" fit density model """
model = KernelMixtureNetwork("KDE_demo", ndim_x=1, ndim_y=1, n_centers=50,
                             x_noise_std=0.2, y_noise_std=0.1, random_seed=22)
model.fit(X, Y)

""" query the conditional pdf and cdf """
x_cond = np.zeros((1, 1))
y_query = np.ones((1, 1)) * 0.1
prob = model.pdf(x_cond, y_query)
cum_prob = model.cdf(x_cond, y_query)

""" compute conditional moments & VaR  """
mean = model.mean_(x_cond)[0][0]
std = model.std_(x_cond)[0][0]
skewness = model.skewness(x_cond)[0]