# DRLearner

## Overview

The **Doubly Robust Learner (DRLearner)** is a meta-algorithm for estimating **heterogeneous treatment effects (CATE)**—how the effect of a treatment varies across individuals—in causal inference.

It leverages both **propensity modeling** and **outcome regression**, combining them into a robust estimate that remains valid if **either model is specified correctly**.

---

## Problem Setup: Potential Outcomes & CATE

Before diving into the `DRLearner` algorithm, it helps to formalize the causal quantities we aim to estimate.

We work in the **Neyman–Rubin potential outcomes framework**, where each sample $i$ has:

- **Covariates:** $X_i \in \mathbb{R}^d$
- **Treatment indicator:** $T_i \in \{0, 1\}$
- **Potential outcomes:**
  - $Y_i(0)$: outcome under control  
  - $Y_i(1)$: outcome under treatment  

Only one outcome is observed:

$$
Y_i^{obs} = Y_i(T_i)
$$

---

### Conditional Response Functions

We define the expected outcome under each treatment level, conditional on covariates:

$$
\mu_j(x) = \mathbb{E}[Y(j) \mid X = x], \quad j = 0,1
$$

These functions summarize how outcomes behave for individuals "like" $x$ under control vs treatment.

---

### Conditional Average Treatment Effect (CATE)

The **treatment effect** for covariate profile $x$ is:

$$
\tau(x) = \mu_1(x) - \mu_0(x)
$$

This quantity tells us **how much the treatment changes the expected outcome** for individuals with features $x$.  
It is the primary object of interest when we care about **heterogeneous treatment effects**—that is, when treatment benefits differ across subgroups or individuals.

---

### Propensity Score

Treatment assignment may be influenced by covariates.  
The **propensity score** captures this:

$$
e(x) = \mathbb{E}[T \mid X = x]
$$

This is the probability that an individual with features $x$ receives the treatment, and it plays a crucial role in adjusting for confounding.

---

### ATE vs. CATE — Why Do We Care About $\tau(x)$?

The **Average Treatment Effect (ATE)**,

$$
\tau = \mathbb{E}[\tau(X)],
$$

is a single number describing the overall benefit of treatment in the population.  
While useful, it hides important individual or subgroup differences.

In many real applications—medicine, genomics, public policy—we want more than the average effect. We want to know:

- *Who benefits most from treatment?*  
- *Who does not benefit?*  
- *Does treatment effectiveness depend on patient characteristics?*

These questions are answered by the **Conditional Average Treatment Effect (CATE)**.

The CATE tells us:

- *not just whether a treatment works*,  
- *but for whom it works* and
- *how its effect varies across the population*.

This is why algorithms such as **DRLearner focus on estimating $\tau(x)$** rather than just the average effect: they aim to uncover **treatment effect heterogeneity**, enabling precision decision-making.

---

## Getting Started

Training a `DRLearner` model is simple in Python.

### Installation

```bash
pip install econml scikit-learn
```

### Initialization Example

```python
from econml.dr import DRLearner
from sklearn.linear_model import LogisticRegression, Ridge

est = DRLearner(
    model_propensity=LogisticRegression(solver='lbfgs', max_iter=200),
    model_regression=Ridge(),
    model_final=Ridge(),
    featurizer=None
)
```

---

## Components

### 1. Propensity Model
- **Example:** `model_propensity=LogisticRegression(...)`
- **Purpose:** Estimate treatment assignment probability.
- **Equation:**

$$
\hat{e}(X_i) = P(T_i = 1 \mid X_i)
$$

---

### 2. Outcome Regression Model

- **Example:** `model_regression=Ridge()`
- **Purpose:** Predict expected outcomes for treated and untreated.
- **Equations:**

$$
\hat{\mu}_1(X_i) = E[Y \mid T=1, X_i]
$$

$$
\hat{\mu}_0(X_i) = E[Y \mid T=0, X_i]
$$

---

### 3. Final Model

- **Example:** `model_final=Ridge()`
- **Purpose:** Fit a model that predicts **estimated treatment effects** using pseudo-outcomes.
- **Equations:** Treat pseudo-outcome $\tilde{Y}_i$ as a regression target and learn a function $\hat{\tau}(x)$ such that:

$$
\hat{\tau}(X_i)\approx\tilde{Y}_i
$$

---

## The DRLearner Equation

For each sample $i$:

1. **Propensity Score:**

$$
\hat{e}_i = P(T_i = 1 \mid X_i)
$$

2. **Potential Outcomes:**
   - $\hat{\mu}_1(X_i)$: Predicted if treated  
   - $\hat{\mu}_0(X_i)$: Predicted if control

3. **Pseudo-outcome Construction:**

$$
\tilde{Y}_i = \left( \frac{T_i - \hat{e}_i}{\hat{e}_i (1 - \hat{e}_i)} \right) \left( Y_i - \hat{\mu}_{T_i}(X_i) \right) + \left( \hat{\mu}_1(X_i) - \hat{\mu}_0(X_i) \right)
$$

4. **Final Step:** Regress $\tilde{Y}_i$ on $X_i$ → CATE.

---

## Summary Table

| Component          | Model                | Role                                      | Output                  |
| ------------------ | -------------------- | ----------------------------------------- | ----------------------- |
| Propensity Model   | `LogisticRegression` | Estimate treatment probability            | Propensity score        |
| Outcome Regression | `Ridge`              | Predict treated/untreated outcomes        | Counterfactual outcomes |
| Final Model        | `Ridge`              | Learn treatment effect via pseudo-outcome | Estimated CATE          |

---

## Key Properties

- **Doubly Robust:** Consistent if *either* the propensity or outcome model is __correctly specified__.  
- **Heterogeneity-aware:** Learns how treatment effects vary with covariates.  
- **Orthogonality:** Uses residualization to stabilize estimation and reduce bias.

---

## FAQs

### Why is it called a meta-algorithm?
Because it **wraps multiple models**—propensity, outcome regression, and final-stage regression—and orchestrates them to produce causal estimates.

### Difference between ATE and CATE?
- **ATE:** One number summarizing overall effect.  
- **CATE:** A function describing how the effect changes with $X$.

### Why does residualization help?
It removes outcome trends explained by covariates, isolating causal effects and improving robustness.

### What are nuisance functions?
Intermediate models (propensity and outcome regression) required for causal effect estimation but not directly of interest.

### Why use pseudo-outcomes?
They de-bias the causal estimate and reduce variance, enabling consistent estimation.

### What does "doubly robust" mean?
Even if one model (propensity or outcome) is misspecified, DRLearner remains **consistent** as long as the other is correct.

### What does it mean for a model to be __correctly specified__?
A model is correctly specified if it captures the true relationship it is meant to estimate. Correct specification does not require perfect accuracy — just that one model captures the correct functional form well enough to avoid bias.

---

## References
- [EconML Documentation](https://econml.azurewebsites.net/)  
- Athey, S., Imbens, G. (2016). *Recursive partitioning for heterogeneous causal effects*. PNAS.  
- Chernozhukov, V., et al. (2018). *Double/debiased machine learning for treatment and structural parameters*.  

---

