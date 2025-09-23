# DRLearner

## 📖 Overview

The **Doubly Robust Learner (DRLearner)** is a meta-algorithm for estimating **heterogeneous treatment effects (CATE)**—how the effect of a treatment varies across individuals—in causal inference.  

It leverages both **propensity modeling** and **outcome regression**, combining them into a robust estimate that remains valid if **either model is specified correctly**.

---

## 🚀 Getting Started

### Installation

You’ll need [econml](https://github.com/microsoft/EconML) and scikit-learn:

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
- **Purpose:** Learns the **Conditional Average Treatment Effect (CATE)** via pseudo-outcomes.
- **Equation:**
  $$
  \hat{\tau}(X_i) = \text{CATE for sample } i
  $$

---

## 🧮 The DRLearner Equation

For each sample \(i\):

1. **Propensity Score:**
   $$
   \hat{e}_i = P(T_i = 1 \mid X_i)
   $$

2. **Potential Outcomes:**
   - \( \hat{\mu}_1(X_i) \): Predicted if treated  
   - \( \hat{\mu}_0(X_i) \): Predicted if control

3. **Pseudo-outcome Construction:**
   $$
   \tilde{Y}_i = \left( \frac{T_i - \hat{e}_i}{\hat{e}_i (1 - \hat{e}_i)} \right) \left( Y_i - \hat{\mu}_{T_i}(X_i) \right) + \left( \hat{\mu}_1(X_i) - \hat{\mu}_0(X_i) \right)
   $$

4. **Final Step:** Regress \(\tilde{Y}_i\) on \(X_i\) → CATE.

---

## 📊 Summary Table

| Component          | Model                | Role                                      | Output                  |
| ------------------ | -------------------- | ----------------------------------------- | ----------------------- |
| Propensity Model   | `LogisticRegression` | Estimate treatment probability            | Propensity score        |
| Outcome Regression | `Ridge`              | Predict treated/untreated outcomes        | Counterfactual outcomes |
| Final Model        | `Ridge`              | Learn treatment effect via pseudo-outcome | CATE estimate           |

---

## 🌟 Key Properties

- **Doubly Robust:** Consistent if *either* propensity or outcome model is correctly specified.  
- **Heterogeneity:** Estimates CATE, not just average treatment effects.  
- **Orthogonality:** Residualization ensures robustness and variance reduction.  

---

## ❓ FAQs

### Why is it a meta-algorithm?
It **wraps other models** for propensity, outcome regression, and CATE learning. It orchestrates them to produce causal estimates.

---

### Difference between ATE and CATE?
- **ATE:** Average effect over the population.  
- **CATE:** Conditional effect given covariates—captures **heterogeneity**.

---

### Why does residualization help?
It removes outcome trends explained by covariates, isolating causal effects and improving robustness.

---

### What are nuisance functions?
Intermediate models (propensity, outcome regression) needed to estimate causal effects, but not of direct interest.

---

### Why staged fitting with pseudo-outcomes?
It de-biases the causal estimate and reduces variance, ensuring consistency.

---

### What does "doubly robust" mean?
Even if one model (propensity or outcome) is misspecified, DRLearner remains **consistent** as long as the other is correct.

---

## 📌 References
- [EconML Documentation](https://econml.azurewebsites.net/)  
- Athey, S., Imbens, G. (2016). *Recursive partitioning for heterogeneous causal effects*. PNAS.  
- Chernozhukov, V., et al. (2018). *Double/debiased machine learning for treatment and structural parameters*.  

---
