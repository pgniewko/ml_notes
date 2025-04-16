# Derivation of the posterioris for the Thompson Sampling

These notes are inspired by Patrick Walters's [paper](https://pubs.acs.org/doi/10.1021/acs.jcim.3c01790) on the application of Thompson Sampling (TS) to virtual high-throughput screening (VHTS).

Our goal is to derive equations (4) and (5) from the paper using a Bayesian update for a univariate Gaussian model with a Gaussian prior.
In the paper, the formulas are provided for each reagent, indexed by $i$.
Since these updates are independent, we drop the reagent index for simplicity and use the index $i$ to denote samples.

## We consider the following setup:
- **Data Model (Likelihood):**  
Assume we have a set of independent and identically distributed (i.i.d.) observations

$$
 \textbf{x} = \(\{x_1, x_2, \dots, x_n\}\)
$$

where each observation is drawn from a normal distribution with unknown mean $\mu$ but known variance $\sigma^2$:

$$
x_i \sim \mathcal{N}(\mu, \sigma^2).
$$

- **Prior:**
The prior on $\mu$ is assumed to be Gaussian:

$$
\mu \sim \mathcal{N}(\mu_0, \sigma_0^2),
$$

where $\mu_0$ is the prior mean and $\sigma_0^2$ is the prior variance.
Our goal is to derive the posterior distribution $p(\mu \mid x_1, \dots, x_n)$ and identify the formulas for the updated mean (posterior mean) and variance (posterior variance).

In the original paper, the variance $\sigma^2$, the initial variance $\sigma_0^2$, and the prior mean $\mu_0$ are estimated from a warmup sample. For a single reagent, we have $\sigma^2 = \sigma_0^2$, whereas for multiple reagent reactions, the priors are estimated from subsamples of each reagent.

Note: During the Bayesian updates, only $\mu_0$ and $\sigma_0^2$ are updated, while $\sigma^2$ remains fixed.

---

## Step 1: Formulate the Likelihood

Given the observations, the likelihood function is

$$
p(\textbf{x} \mid \mu) = \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x_i - \mu)^2}{2\sigma^2}\right).
$$

Taking the logarithm (and ignoring constants that do not depend on $\mu$):

$$
\log p(\textbf{x} \mid \mu) \propto -\frac{1}{2\sigma^2}\sum_{i=1}^{n}(x_i - \mu)^2.
$$

Expanding the square inside the sum gives:

$$
\sum_{i=1}^{n} (x_i - \mu)^2 = \sum_{i=1}^{n} x_i^2 - 2\mu\sum_{i=1}^{n} x_i + n\mu^2.
$$

Thus, in terms of $\mu$, the likelihood’s exponent becomes:

$$
-\frac{1}{2\sigma^2}\left(n\mu^2 - 2\mu\sum_{i=1}^{n} x_i\right) + \text{(terms independent of $\mu$)}.
$$

---

## Step 2: Write Down the Prior

The prior distribution for $\mu$ is

$$
p(\mu) = \frac{1}{\sqrt{2\pi\sigma_0^2}} \exp\left(-\frac{(\mu - \mu_0)^2}{2\sigma_0^2}\right).
$$

Taking the logarithm (and focusing on terms involving $\mu$):

$$
\log p(\mu) \propto -\frac{1}{2\sigma_0^2} \left(\mu^2 - 2\mu\mu_0\right).
$$

---

## Step 3: Combine Likelihood and Prior to Form the Posterior

Using Bayes’ rule, the posterior is proportional to the product of the likelihood and the prior:

$$
p(\mu \mid \textbf{x}) \propto p(\textbf{x} \mid \mu) \, p(\mu).
$$

So, we add the logarithms from the likelihood and the prior:

$$
\log p(\mu \mid \textbf{x}) \propto -\frac{1}{2\sigma^2}\left(n\mu^2 - 2\mu\sum_{i=1}^{n} x_i\right) - \frac{1}{2\sigma_0^2}\left(\mu^2 - 2\mu\mu_0\right).
$$

Grouping the quadratic and linear terms in $\mu$:

$$
\log p(\mu \mid \textbf{x}) \propto -\frac{1}{2} \left[ \mu^2\left(\frac{n}{\sigma^2} + \frac{1}{\sigma_0^2}\right) - 2\mu \left(\frac{\sum_{i=1}^{n} x_i}{\sigma^2} + \frac{\mu_0}{\sigma_0^2}\right) \right].
$$

This expression is quadratic in $\mu$ and it represents the logarithm of a normal density - Gaussian distribution is a conjugate prior on $\mu$. To read off the parameters, it is most instructive to "complete the square".

---

## Step 4: Complete the Square

We now re-write the quadratic expression in $\mu$ to match the standard form of a Gaussian exponent. Define

$$
A = \frac{n}{\sigma^2} + \frac{1}{\sigma_0^2}, \quad \text{and} \quad B = \frac{\sum_{i=1}^{n} x_i}{\sigma^2} + \frac{\mu_0}{\sigma_0^2}.
$$

The log-posterior can be rewritten as:

$$
\log p(\mu \mid \textbf{x}) \propto -\frac{A}{2}\left(\mu^2 - 2\mu\frac{B}{A}\right).
$$

Completing the square yields:

$$
-\frac{A}{2}\left(\mu - \frac{B}{A}\right)^2 + \text{constant}.
$$

Thus, the posterior distribution is a Gaussian given by:

$$
\mu \mid \textbf{x} \sim \mathcal{N}\left(\mu_{n}, \sigma_{n}^2\right)
$$

with:

- **Posterior Variance:**
  
$$
  \sigma_n^2 = \frac{1}{A} = \left(\frac{n}{\sigma^2} + \frac{1}{\sigma_0^2}\right)^{-1} \overset{\text{eq. 5}}{=} \frac{\sigma_0^2\sigma^2}{\sigma^2 + n\sigma_0^2}
$$

- **Posterior Mean:**

$$
  \mu_n = \frac{B}{A} = \sigma_n^2 \left(\frac{\sum_{i=1}^{n} x_i}{\sigma^2} + \frac{\mu_0}{\sigma_0^2}\right) = \left(\frac{\sigma_0^2\sigma^2}{\sigma^2 + n\sigma_0^2}\right)\left(\frac{n\bar{x}\sigma_0^2 + \sigma^2\mu_0}{\sigma^2\sigma_0^2}\right) \overset{\text{eq. 4}}{=} \frac{\sigma^2\mu_0 + n\bar{x}\sigma_0^2}{\sigma^2 + n\sigma_0^2}
$$

Were we expressed $\sum_{i=1}^{n} x_i$ as $n\bar{x}$.

---

## Step 5: Interpretation

- **Posterior Variance $\sigma_n^2$:**  
  This quantity represents the updated uncertainty about the mean $\mu$. Notice that as the number of data points $n$ increases, the term $\frac{n}{\sigma^2}$ grows, leading to a decrease in $\sigma_n^2$. This reflects the idea that more data yields a more precise estimate of $\mu$.

- **Posterior Mean $\mu_n$:**  
  The updated mean is a weighted average of the sample mean $\bar{x}$ and the prior mean $\mu_0$. The weights are determined by the relative precisions (inverse variances) $\frac{1}{\sigma^2}$ and $\frac{1}{\sigma_0^2}$. More weight is given to the component (data or prior) that is more "precise" (i.e., has a smaller variance).

---

## Summary of the Updates

- **Posterior Variance:**

$$
  \sigma_n^2 = \left(\frac{n}{\sigma^2} + \frac{1}{\sigma_0^2}\right)^{-1}.
$$

- **Posterior Mean:**

$$
  \mu_n = \sigma_n^2 \left(\frac{n\bar{x}}{\sigma^2} + \frac{\mu_0}{\sigma_0^2}\right).
$$

These formulas are central in Bayesian inference when using conjugate priors, and they can also be applied sequentially: by treating the posterior from one update as the new prior for the next update.

---

## How to Use This in Practice

1. **Set Up Your Model:**  
   Define your likelihood as $x_i \sim \mathcal{N}(\mu, \sigma^2)$ and your prior as $\mu \sim \mathcal{N}(\mu_0, \sigma_0^2)$.

2. **Collect Data:**  
   Gather your observations $\{x_1, \dots, x_n\}$ and compute their sum (or sample mean $\bar{x}$).

3. **Calculate the Posterior Parameters:**  
   Use the formulas above to compute $\sigma_n^2$ and $\mu_n$.

4. **Update Your Beliefs:**  
   Use the posterior $\mathcal{N}(\mu_n, \sigma_n^2)$ as your updated belief about $\mu$, which can now serve as the new prior if additional data arrives.
