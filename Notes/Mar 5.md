[TOC]

笛卡尔积，两个集合的乘法

multivariate gaussian distribiution

$x\sim N(0,1)$

$(x_1,x_2)\sim N(\boldsymbol{0},I)$

$\boldsymbol{0}=[0,0]$

$I=$

# Learn a distribution

Observed data is $\{x_1,x_2,\ldots,x_n\}$, $p(x)$

## Traditional Statistical Methods

### Parametric methods

There is a pre-determined distribution, for example, normal distribution $\mathcal{N}(\mu,\sigma)$ with density funciton $f_{\mu,\sigma}(\cdot)$.

The likelihood of observing $\{x_1,x_2,\ldots,x_n\}$ is $\Pi_{i=1}^n f_{\mu,\sigma}(x_i)$.

Log-likelihood is $\sum_{i=1}^n \log f_{\mu,\sigma}(x_i)$.

The parameters $\mu,\sigma$ need to be optimized by maximizing $\sum_{i=1}^n \log f_{\mu,\sigma}(x_i)$.

### Non-parametric methods

## Machine Learning methods

Generative adversiral networks GAN

Variational Auto Encoder VAE

Normalizing Flow

# Learn a conditional distribution

Observed data is $\{(x_1,y_1),(x_2,y_2),\ldots,(x_n,y_n)\}$. These samples come from a underlying distribution $p(x|y)$.

## Traditional Statistical Methods

### Parametric methods

### Non-parametric methods



## Machine Learning methods

# Evaluation Metrics

在conditional distribution的setting下，如何评估learn的conditional distribution与真实的$p(x|y)$之间的“距离”

用来衡量两个分布之间的距离的一些metric：

1. KL divergence

2. Wasserstein Distance

# Homework

## Experiments and datasets

Conditional distribution相关的文章里，大家都做了什么实验，用了什么样的数据集，怎样评估实验的效果（用了什么evaluation metric，比如KL divergence）

对于数据集，了解一下有多少样本，样本的condition的维度和dependent的维度是多少



