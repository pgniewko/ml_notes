# PC Algorithm & Conditional Independence Tests — Comprehensive Tutorial

This tutorial is a comprehensive guide to the **Peter–Clark (PC) algorithm** for causal discovery, the **fundamental concepts** it builds upon, and the **methods used to test conditional independence (CI)**. We will summarize these methods, present their implementations, and compare their performance — especially for continuous variables with nonlinear relationships.

---

## 1. Fundamental Concepts

### 1.1 Directed Acyclic Graph (DAG)
A DAG is a graph with directed edges and no cycles, often used to represent causal structures.

### 1.2 Causal Markov Condition
In a causal DAG, each variable is independent of its **non-descendants** given its **parents**.

**Example:** In `A → B → C`, C is independent of A given B.

### 1.3 Faithfulness
The only CI relations present in the data are those implied by the DAG’s structure. No independencies arise purely from coincidental parameter cancellations.

**Example:** In `A → B → C`, if parameters cancel so A and C appear independent, faithfulness is violated.

### 1.4 d-Separation
A graphical criterion to determine if two nodes are conditionally independent given a set of other nodes.

### 1.5 Colliders (V-Structures)
A triple `A → B ← C` is a collider if two arrowheads meet at B and A, C are not adjacent.

**PC Orientation Rule:** If in `A - B - C`, A and C are nonadjacent and B is **not** in the separating set of A and C, orient as `A → B ← C`.

### 1.6 Meek’s Rules
After orienting colliders, **Meek’s rules** propagate orientations without introducing cycles or new, unsupported colliders:

- **R1:** If `A → B - C` and A, C nonadjacent, orient `B → C`.
- **R2:** If `A - B` and there exists `A → C → B`, orient `A → B`.
- **R3:** If `A - B`, `A - C`, `C → B`, `A → C`, orient `A → B`.
- **R4:** If `A - B` and there exist Z1, Z2 with `A - Z1`, `Z1 → B`, `A - Z2`, `Z2 → B`, and Z1, Z2 nonadjacent, orient `A → B`.

### 1.7 Markov Equivalence Class & CPDAG
Two DAGs are Markov equivalent if they share the same skeleton and v-structures. Observational data cannot distinguish them. The PC algorithm outputs a **CPDAG** representing all DAGs in the equivalence class.

---

## 2. PC Algorithm Steps

1. **Initialize:** Complete undirected graph.
2. **Skeleton learning:** Iteratively remove edges using CI tests, starting with unconditional tests, then conditioning sets of size 1, 2, etc.
3. **Collider orientation:** Identify and orient v-structures.
4. **Apply Meek’s rules:** Propagate orientations until no more can be made.

---

## 3. Conditional Independence Testing Methods

### 3.1 Discrete Data

#### G-Test (Likelihood-Ratio Test)
**Formula:**  
\[ G = 2 \sum_{i,j} O_{ij} \log \frac{O_{ij}}{E_{ij}} \]  
Where O = observed counts, E = expected counts under independence.

**Code:**
```python
import pandas as pd
from scipy.stats import chi2_contingency, chi2

def g_test_discrete(X, Y):
    ct = pd.crosstab(X, Y)
    G, p, dof, _ = chi2_contingency(ct, lambda_="log-likelihood")
    return G, p, dof

def g_test_conditional(X, Y, Z):
    df = pd.DataFrame({"X": X, "Y": Y, "Z": Z})
    G_total, df_total = 0, 0
    for _, group in df.groupby("Z"):
        ct = pd.crosstab(group.X, group.Y)
        if ct.shape[0] > 1 and ct.shape[1] > 1:
            G, _, dof, _ = chi2_contingency(ct, lambda_="log-likelihood")
            G_total += G
            df_total += dof
    p_value = 1 - chi2.cdf(G_total, df_total)
    return G_total, p_value, df_total
```

#### Chi-Squared Test
Similar to G-test but uses Pearson’s chi-squared statistic.

---

### 3.2 Continuous Data (Linear Gaussian)

#### Partial Correlation / Fisher Z Test
**Idea:** Regress X and Y on Z, then test the correlation of residuals.

**Code:**
```python
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr, t
import numpy as np

def partial_corr_test(X, Y, Z):
    if Z.ndim == 1: Z = Z.reshape(-1, 1)
    rx = X - LinearRegression().fit(Z, X).predict(Z)
    ry = Y - LinearRegression().fit(Z, Y).predict(Z)
    r, _ = pearsonr(rx.ravel(), ry.ravel())
    n = len(rx)
    dof = n - 2
    t_stat = r * np.sqrt(dof / (1 - r**2))
    p_value = 2 * (1 - t.cdf(abs(t_stat), dof))
    return r, p_value
```

---

### 3.3 Continuous Data (Nonlinear / Non-Gaussian)

#### KCI (Kernel-based Conditional Independence)
Uses kernel embeddings in RKHS; robust to nonlinearities.

#### Conditional Mutual Information (CMI)
Estimated using k-nearest neighbor entropy estimators.

**Code:**
```python
from npeet.entropy_estimators import cmi
X = np.random.randn(200)
Y = np.random.randn(200)
Z = np.random.randn(200)
obs_cmi = cmi(X, Y, Z)
print("CMI:", obs_cmi)
```

---

## 4. Summary Table

| Data Type   | Assumptions             | Method               | Strengths | Weaknesses |
|-------------|------------------------|----------------------|-----------|------------|
| Discrete    | Any                    | G-test / Chi-squared | Simple, fast | Needs enough counts |
| Continuous  | Linear, Gaussian       | Partial correlation  | Easy, fast | Fails on nonlinear/non-Gaussian |
| Continuous  | Nonlinear, Non-Gaussian| KCI                  | General, nonparametric | Slower |
| Continuous  | Nonlinear, Non-Gaussian| CMI (kNN)            | Captures arbitrary deps | Sensitive to k, slow |

---

## 5. Performance Comparison: Continuous, Nonlinear, Non-Gaussian

**Scenario:**  
- Z ~ Uniform(-3,3)  
- X = sin(Z) + Laplace noise  
- Y = cos(Z) + Laplace noise  
- X ⟂ Y | Z in truth.

**Code:**
```python
from scipy.stats import laplace
from pycitest import KCI

np.random.seed(42)
n = 400
Z = np.random.uniform(-3, 3, (n,1))
X = np.sin(Z) + 0.4 * laplace.rvs(size=(n,1))
Y = np.cos(Z) + 0.4 * laplace.rvs(size=(n,1))

# Partial correlation
r, p_lin = partial_corr_test(X, Y, Z)
print("Partial corr:", r, "p:", p_lin)

# KCI
kci = KCI()
stat, p_kci = kci.test(X, Y, Z)
print("KCI stat:", stat, "p:", p_kci)
```

**Expected Outcome:**  
- Partial correlation: small p-value → false dependence (misses nonlinear independence).  
- KCI: large p-value → correct detection of independence.

---
