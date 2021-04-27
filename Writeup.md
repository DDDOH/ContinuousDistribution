**Why learn conditional distribution?**

A wide range of problems in machine learning concerns learning the relationship between a variable $x$ and another variable $y$.  In prediction task, the target is to build a deterministic mapping such that $y=f(x)$.

<img src="https://tva1.sinaimg.cn/large/008i3skNly1gpysgoyd2gj30x80kctae.jpg" alt="image-20210428005452382" style="zoom:25%;" />

However, such deterministic relationship may not exist. Instead, given $X=x$, it is better to model $Y$ as a random variable with a conditional distribution $\mathcal{P}_{Y|_{X=x}}$.

To learn the conditional distribution is also of great interset to

1. Finance
2. Operation research
3. XXX
4. XXX

**Why regularization?**

Prediction can overfitting, so does learning the conditional distribution.

Two figures: the overfitting of prediction task, the overfitting of learning the conditional distribution.

In some applications, the dimension of the condition may be quite large, and the number of available samples are limited. This results in the sparsity of the samples’ location, which makes overfitting more inclined to happen.

**Our approach:**

Notation:

The underlying distribution:
$$
X\in\mathbb{R}^{n_x}, Y\in\mathbb{R}^{n_y},(X,Y)\sim \mathcal{P}_{(X,Y)}, X\sim \mathcal{P}_X,  Y\sim \mathcal{P}_Y, Y|_{X=x}\sim \mathcal{P}_{Y|_{X=x}}
$$
Dataset:

$N$ iid samples of $\{x_i,y_i\}_{i=1}^N$.

We use MDN (Mixture density network) to model the conditional distribution. 

<img src="/Users/shuffle_new/Library/Application Support/typora-user-images/image-20210428013326703.png" alt="image-20210428013326703" style="zoom:33%;" />

This network models the conditional distribution $\mathcal{P}_{Y|_{X=x}}$ using GMM. (Why GMM? universal approximator)

Input of the network: $x$

output of the network: parameters of the GMM: weight list $\alpha_i(\mathbf{x})$, mean $\boldsymbol{\mu}_{i}(\mathbf{x})$, variance $\sigma_{i}(\mathbf{x})^{2}$ for $i=1,2,\ldots,m$. ($m$ is a hyper-parameter).

The (probability density function) pdf of Gaussian mixture model is
$$
p(\mathbf{y}\mid \mathbf{x})=\sum_{i=1}^{m} \alpha_{i}(\mathbf{x}) \phi_{i}(\mathbf{y} \mid \mathbf{x})
$$
with
$$
\phi_{i}(\mathbf{y} \mid \mathbf{x})=\frac{1}{(2 \pi)^{c / 2} \sigma_{i}(\mathbf{x})^{c}} \exp \left\{-\frac{\left\|\mathbf{y}-\boldsymbol{\mu}_{i}(\mathbf{x})\right\|^{2}}{2 \sigma_{i}(\mathbf{x})^{2}}\right\}
$$
We train this network by maximizing the loglikelihood:
$$
\text{Log Likelihood} = \sum_{i=1}^N\ln \left\{\sum_{i=1}^{m} \alpha_{i}\left(\mathbf{x}_i\right) \cdot \phi_{i}\left(\mathbf{y}_{i} \mid \mathbf{x}_{i}\right)\right\}
$$
The regularization term:





$$

$$

$$
X=x, D(Y_{real}|_{X=x},Y_{network}|_{X=x}), E_{X\sim \mathcal{P}_X}[D(Y_{real}|_{X},Y_{network}|_{X})]
$$

