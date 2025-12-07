# DRLearner

## 📖 Overview

The **Doubly Robust Learner (DRLearner)** is a meta-algorithm for estimating **heterogeneous treatment effects (CATE)**—how the effect of a treatment varies across individuals—in causal inference.

It leverages both **propensity modeling** and **outcome regression**, combining them into a robust estimate that remains valid if **either model is specified correctly**.

---

## 🧩 Problem Setup: Potential Outcomes & CATE

Before diving into the DRLearner algorithm, it helps to formalize the causal quantities we aim to estimate.

We work in the **Neyman–Rubin potential outcomes framework**, where each sample \(i\) has:

- **Covariates:** \(X_i \in \mathbb{R}^d\)
- **Treatment indicator:** \(T_i \in \{0, 1\}\)
- **Potential outcomes:**
  - \(Y_i(0)\): outcome under control  
  - \(Y_i(1)\): outcome under treatment  

Only one outcome is observed:

\[
Y_i^{	ext{obs}} = Y_i(T_i)
\]

### Response functions

\[
\mu_j(x) = \mathbb{E}[Y(j)\mid X = x], \quad j \in \{0,1\}
\]

### Conditional Average Treatment Effect (CATE)

\[
	au(x) = \mu_1(x) - \mu_0(x)
\]

The CATE describes **how the treatment effect varies depending on features**.

### Propensity score

\[
e(x) = \mathbb{E}[T \mid X = x]
\]

### ATE vs CATE — Why CATE Matters

The **Average Treatment Effect (ATE)**,

\[
	au = \mathbb{E}[	au(X)],
\]

summarizes the effect of treatment across the entire population.  
However, people are not average—treatment effects often vary widely across subgroups or individuals.

- ATE answers: **“Does the treatment help on average?”**
- CATE answers: **“Who does the treatment help, and by how much?”**

**DRLearner focuses on estimating CATE**, enabling personalized medicine, subgroup analysis, and precision decision-making.

---

## 🚀 Getting Started

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

## 🔎 Components

### 1. Propensity Model
- **Example:** `model_propensity=LogisticRegression(...)`
- **Purpose:** Estimate treatment assignment probability.
- **Equation:**

\[
\hat{e}(X_i) = P(T_i = 1 \mid X_i)
\]

---

### 2. Outcome Regression Model

- **Example:** `model_regression=Ridge()`
- **Purpose:** Predict expected outcomes for treated and untreated.
- **Equations:**

\[
\hat{\mu}_1(X_i) = E[Y \mid T=1, X_i]
\]

\[
\hat{\mu}_0(X_i) = E[Y \mid T=0, X_i]
\]

---

### 3. Final Model

- **Example:** `model_final=Ridge()`
- **Purpose:** Fit a model that predicts **estimated treatment effects** using pseudo-outcomes.
- **What it does:**  
  Treat \(	ilde{Y}_i\) as a regression target and learn a function \(\hat{	au}(x)\) such that:

\[
\hat{	au}(X_i) pprox 	ilde{Y}_i
\]

- **Interpretation:**  
  \(\hat{	au}(x)\) is our estimate of how treatment effects vary with covariates \(x\).  
  It is not the true CATE but an approximation based on the data and nuisance models.

---

## 🧮 The DRLearner Equation

For each sample \(i\):

### 1. Propensity Score:

\[
\hat{e}_i = P(T_i = 1 \mid X_i)
\]

### 2. Potential Outcomes:

- \(\hat{\mu}_1(X_i)\): Predicted outcome if treated  
- \(\hat{\mu}_0(X_i)\): Predicted outcome if control

### 3. Pseudo-outcome Construction:

\[
	ilde{Y}_i =
\left(rac{T_i - \hat{e}_i}{\hat{e}_i (1 - \hat{e}_i)}ight)\left(Y_i - \hat{\mu}_{T_i}(X_i)ight)
+ \left(\hat{\mu}_1(X_i) - \hat{\mu}_0(X_i)ight)
\]

### 4. Final Step:

Regress \(	ilde{Y}_i\) on \(X_i\) → estimated CATE.

---

## 📊 Summary Table

| Component          | Model                | Role                                      | Output                  |
| ------------------ | -------------------- | ----------------------------------------- | ----------------------- |
| Propensity Model   | `LogisticRegression` | Estimate treatment probability            | Propensity score        |
| Outcome Regression | `Ridge`              | Predict treated/untreated outcomes        | Counterfactual outcomes |
| Final Model        | `Ridge`              | Learn treatment effect via pseudo-outcome | Estimated CATE          |

---

## 🌟 Key Properties

- **Doubly Robust:** Consistent if *either* the propensity or outcome model is correctly specified.  
- **Heterogeneity-aware:** Learns how treatment effects vary with covariates.  
- **Orthogonality:** Uses residualization to stabilize estimation and reduce bias.

---

## ❓ FAQs

### Why is it called a meta-algorithm?
Because it **wraps multiple models**—propensity, outcome regression, and final-stage regression—and orchestrates them to produce causal estimates.

### Difference between ATE and CATE?
- **ATE:** One number summarizing overall effect.  
- **CATE:** A function describing how the effect changes with \(X\).

### Why does residualization help?
It removes trends explained by covariates, isolating causal signal.

### What are nuisance functions?
Intermediate models (propensity and outcome regression) required for causal effect estimation but not directly of interest.

### Why use pseudo-outcomes?
They de-bias the causal estimate and reduce variance, enabling consistent estimation.

---

## 📌 References

- EconML Documentation  
- Athey & Imbens (2016). *Recursive partitioning for heterogeneous causal effects*.  
- Chernozhukov et al. (2018). *Double/debiased machine learning for treatment and structural parameters*.

