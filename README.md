# EM Algorithm for 1D Gaussian Mixture Model

This tutorial explains the derivation of the Expectation-Maximization (EM) algorithm for fitting a 1D Gaussian mixture model with $N$ components. It covers two main derivation tasks:

1. **Posterior Distribution $P(z|x)$:**  
   Deriving an expression for the posterior distribution over the latent variable $z \in \{1, 2, \dots, N\}$ for a given data point $x$.

2. **Evidence Lower Bound (ELBO) Update:**  
   Deriving an expression for updating the evidence lower bound given the posterior distributions for all data points, including the use of Lagrange multipliers to ensure that the mixing weights $\lambda_1, \ldots, \lambda_N$ sum to one.

In addition, this README describes how the update formulas for the Gaussian parameters—means and variances—are derived within the EM framework.

---

## 1. Derivation of the Posterior Distribution $P(z|x)$ (E-Step)

For a 1D Gaussian mixture model with $N$ components, each component $j$ has:
- A mixing weight: $\lambda_j$; subjecto to constraint $\sum_{j=1}^N \lambda_j = 1$
- A mean: $\mu_j$
- A variance: $\sigma_j^2$

Given a data point $x$, the likelihood under component $j$ is:

$$
p(x|z=j) = \mathcal{N}(x; \mu_j, \sigma_j^2)
$$

and the joint probability is:

$$
p(x, z=j) = \lambda_j  \mathcal{N}(x; \mu_j, \sigma_j^2).
$$

By applying Bayes' rule, the posterior distribution (or "responsibility") is:

$$
P(z=j|x) = \frac{p(x,z=j)}{p(x)} = \frac{\lambda_j \mathcal{N}(x; \mu_j, \sigma_j^2)}{\sum_{k=1}^N \lambda_k \mathcal{N}(x; \mu_k, \sigma_k^2)}.
$$

We denote this as:

$$
\gamma_j(x) \triangleq P(z=j|x).
$$

---

## 2. Updating the Evidence Lower Bound (ELBO) (M-Step)

The EM algorithm maximizes the log-likelihood by maximizing the Evidence Lower Bound (ELBO). For a single data point $x$ and a posterior $q(z)$ (which is chosen as $P(z|x))$, the ELBO is given by:

$$
\mathcal{L}(q,\theta) = \sum_{z} q(z) \log \frac{P(x,z|\theta)}{q(z)}.
$$

For a dataset $\{x}_{i=1}^M$, the total ELBO is:

$$
\mathcal{L} = \sum_{i=1}^M \sum_{j=1}^N \gamma_j(x_i) \log \frac{\lambda_j \mathcal{N}(x_i; \mu_j, \sigma_j^2)}{\gamma_j(x_i)} = \sum_{i=1}^M \sum_{j=1}^N \gamma_{ij} \log \frac{\lambda_j \mathcal{N}(x_i; \mu_j, \sigma_j^2)}{\gamma_j(x_i)}
$$

where we replaced $\gamma_j(x_i) = \gamma_{ij}$ for brevity.

### 2.1. Updating the Mixing Weights $\lambda_j$

Only the term:

$$
\sum_{i=1}^M \gamma_{ij} \log \lambda_j
$$

depends on the mixing weights. To update $\lambda_j$, we maximize:

$$
\mathcal{L_\lambda} = \sum_{i=1}^M \sum_{j=1}^N \gamma_{ij} \log \lambda_j.
$$

subject to the constraint:

$$
\sum_{j=1}^N \lambda_j = 1.
$$

#### Using Lagrange Multipliers

Define the Lagrangian:

$$
\mathcal{J} = \sum_{i=1}^M \sum_{j=1}^N \gamma_{ij} \log \lambda_j + \alpha \left(1 - \sum_{j=1}^N \lambda_j\right).
$$

Taking the derivative with respect to $\lambda_j$ and setting it to zero:

$$
\frac{\partial \mathcal{J}}{\partial \lambda_j} = \frac{1}{\lambda_j} \sum_{i=1}^M \gamma_{ij} - \alpha = 0,
$$

which yields:

$$
\lambda_j = \frac{1}{\alpha} \sum_{i=1}^M \gamma_{ij}.
$$

Enforcing the constraint:

$$
\sum_{j=1}^N \lambda_j = \frac{1}{\alpha} \sum_{i=1}^M \sum_{j=1}^N \gamma_{ij} = \frac{1}{\alpha} M = 1,
$$

we find $\alpha = M$. Therefore, the updated mixing weights are:

$$
\lambda_j = \frac{1}{M} \sum_{i=1}^M \gamma_j(x_i).
$$

---

## 3. Derivation of the Updates for the Means and Variances

### 3.1. The Complete-Data Log-Likelihood $Q(\theta)$

For a Gaussian mixture model, the complete-data log-likelihood is:

$$
\log p(x \mid \theta) = \sum_{i=1}^M \sum_{j=1}^N I(z_i = j) \left[\log \lambda_j + \log \mathcal{N}(x_i; \mu_j, \sigma_j^2)\right],
$$

where $I(z_i = j)$ is an indicator function.

Since $z$ is unobserved, we take the expectation with respect to the posterior $P(z_i=j|x_i)$ (i.e., using responsibilities $\gamma_{ij}\$):

$$
Q(\theta) = \sum_{i=1}^N \sum_{j=1}^N \gamma_{ij} \left[\log \lambda_j + \log \mathcal{N}(x_i; \mu_j, \sigma_j^2)\right].
$$

The Gaussian density term is:

$$
\log \mathcal{N}(x_i; \mu_j, \sigma_j^2) = -\frac{1}{2}\log(2\pi \sigma_j^2) - \frac{(x_i-\mu_j)^2}{2\sigma_j^2}.
$$

### 3.2. Updating the Mean $\mu_j$

Focus on the term involving $\mu_j$:

$$
-\frac{1}{2\sigma_j^2}\sum_{i=1}^N \gamma_{ij}(x_i-\mu_j)^2.
$$

Taking the derivative with respect to $\mu_j$ and setting it to zero:

$$
\frac{\partial}{\partial \mu_j} \left[-\frac{1}{2\sigma_j^2}\sum_{i=1}^N \gamma_{ij}(x_i-\mu_j)^2\right] = \frac{1}{\sigma_j^2}\sum_{i=1}^N \gamma_{ij}(x_i-\mu_j) = 0.
$$

Solving for $\mu_j$ gives:

$$
\mu_j = \frac{\sum_{i=1}^N \gamma_{ij} x_i}{\sum_{i=1}^N \gamma_{ij}}.
$$

### 3.3. Updating the Variance $\sigma_j^2$

Consider the terms involving variance $\sigma_j^2$:

$$
\sum_{i=1}^N \gamma_{ij}\left[-\frac{1}{2}\log(2\pi \sigma_j^2) - \frac{(x_i-\mu_j)^2}{2\sigma_j^2}\right].
$$

Taking the derivative with respect to $\sigma_j^2$ and setting it to zero:

$$
-\frac{1}{2\sigma_j^2}\sum_{i=1}^N \gamma_{ij} + \frac{1}{2(\sigma_j^2)^2}\sum_{i=1}^N \gamma_{ij}(x_i-\mu_j)^2 = 0.
$$

Multiplying both sides by $2(\sigma_j^2)^2$ leads to:

$$
-\sigma_j^2\sum_{i=1}^N \gamma_{ij} + \sum_{i=1}^N \gamma_{ij}(x_i-\mu_j)^2 = 0.
$$

Thus, the update for the variance is:

$$
\sigma_j^2 = \frac{\sum_{i=1}^N \gamma_{ij}(x_i-\mu_j)^2}{\sum_{i=1}^N \gamma_{ij}}.
$$

---

## Summary

### EM Algorithm Steps

- **E-Step:**  
  Compute the posterior (responsibilities) for each data point $x$ and component $j$:

$$
  \gamma_j(x) = \frac{\lambda_j \, \mathcal{N}(x; \mu_j, \sigma_j^2)}{\sum_{k=1}^N \lambda_k \, \mathcal{N}(x; \mu_k, \sigma_k^2)}.
$$

- **M-Step:**  
  Update the parameters using the computed responsibilities:
   - **Mixing Weights:**

$$
   \lambda_j = \frac{1}{M} \sum_{i=1}^M \gamma_{ij}.
$$

   - **Means:**

$$
   \mu_j = \frac{\sum_{i=1}^N \gamma_{ij} x_i}{\sum_{i=1}^N \gamma_{ij}}.
$$

   - **Variances:**

$$
   \sigma_j^2 = \frac{\sum_{i=1}^N \gamma_{ij}(x_i-\mu_j)^2}{\sum_{i=1}^N \gamma_{ij}}.
$$

These derivations ensure that each update step increases the likelihood (or, equivalently, the ELBO) and handles the latent variable assignments effectively via the responsibilities. The use of Lagrange multipliers in updating the mixing weights guarantees that: $\sum_{j=1}^N \lambda_j = 1\.$
