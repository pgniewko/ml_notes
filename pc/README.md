# PC Algorithm & Conditional Independence Tests — Tutorial README

This README is for a tutorial on the **Peter–Clark (PC) algorithm** for causal discovery. It introduces key concepts like **Faithfulness**, **Causal Markov Condition**, **collider identification**, and **Meek rules**, then focuses on **conditional independence (CI) testing** for discrete and continuous variables, with runnable Python examples.

---

## Key Concepts

### Causal Markov Condition
**Definition:** In a DAG, each node is conditionally independent of its non-descendants given its parents.  
**Example:** If `A → B → C`, then C is independent of A given B.  
Formally: if `Pa(X)` denotes parents of X, then  
\[ X ⟂ NonDesc(X) \mid Pa(X) \]

---

### Faithfulness Condition
**Definition:** All and only the conditional independencies in the data come from the DAG's d-separation.  
**Example:** If `A → B → C` with parameters such that the effect of A cancels exactly with noise, A and C might appear independent — this violates faithfulness.  
**Why it matters:** PC relies on statistical CI to reflect graph structure; violations can cause wrong edge deletions/orientations.

---

### Colliders and Collider Orientation Rule
**Pattern:** A **v-structure** `A → B ← C` where A and C are not connected.  
**Orientation logic:** If in an unshielded triple `A - B - C`, we find that B is **not** in the separating set for A and C, we orient as `A → B ← C`.  
**Example:** If `A` and `C` are independent given `{}`, but dependent given `B`, B is a collider.  

---

### Meek Rules (Comprehensive)
Rules to propagate orientations without introducing cycles or unsupported colliders:

- **R1:** If `A → B - C` and A and C are nonadjacent, then orient `B - C` as `B → C`.
- **R2:** If `A - B` and there exists `A → C → B`, then orient `A - B` as `A → B`.
- **R3:** If `A - B`, and there exists `A - C` with `C → B` and `A → C`, then orient `A - B` as `A → B`.
- **R4:** If `A - B` and there exist two nodes Z1, Z2 such that `A - Z1`, `Z1 → B`, `A - Z2`, `Z2 → B`, and Z1 and Z2 are nonadjacent, then `A → B`.
- **R5 (sometimes included):** Orient edges to avoid creating new v-structures unless supported by CI tests.

---

## Markov Equivalence Class & Observational Data Limitations
Two DAGs are **Markov equivalent** if they have the **same skeleton** (adjacencies) and **same v-structures**.  
**Problem:** Observational data alone cannot distinguish between DAGs in the same class.  
**Example:** `A → B → C` and `A ← B → C` are Markov equivalent (same skeleton, no colliders).  
**PC Output:** A **CPDAG** — partially directed graph representing all DAGs in the equivalence class.

---

## Summary Table: Collider and Meek Rules

| Rule Type | Main Use | Pattern | Orientation Logic |
|-----------|----------|---------|-------------------|
| Collider  | Identify v-structures | `A - B - C`, A and C nonadjacent, B not in Sep(A,C) | Orient `A → B ← C` |
| R1        | Propagation | `A → B - C`, A and C nonadjacent | Orient `B → C` |
| R2        | Avoid new colliders | `A - B` with `A → C → B` | Orient `A → B` |
| R3        | Cycle avoidance | `A - B` and `A - C`, `C → B`, `A → C` | Orient `A → B` |
| R4        | Multiple parents into node | `A - B` and Z1, Z2 as described | Orient `A → B` |

---

## Why Certain Steps in PC Matter
- **Skeleton learning:** Removes spurious edges — crucial for avoiding false dependencies later.
- **Collider orientation:** First guaranteed directions — sets partial order.
- **Meek rules:** Expand orientations logically, reducing ambiguity.
- **Faithfulness assumption:** Links statistical CI to graph structure.

---

## G-Test for Discrete Data
**Definition:** Likelihood-ratio test for independence.  
**Formula:**  
\[ G = 2 \sum_{i,j} O_{ij} \log\left( \frac{O_{ij}}{E_{ij}} \right) \]  
where O = observed counts, E = expected counts under independence.  
**Usage:** Compare G to chi-squared distribution with df = (rows-1)(cols-1).

**Python Example:**
```python
import pandas as pd
from scipy.stats import chi2_contingency, chi2

def g_test_discrete(X, Y):
    ct = pd.crosstab(X, Y)
    G, p, dof, exp = chi2_contingency(ct, lambda_="log-likelihood")
    return G, p, dof

# Conditional version
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

---

## Conditional Mutual Information Example
```python
from npeet.entropy_estimators import cmi
import numpy as np

X = np.random.randn(200)
Y = np.random.randn(200)
Z = np.random.randn(200)

obs_cmi = cmi(X, Y, Z)
print("CMI:", obs_cmi)
```

---

## Nonlinear Dependency & Non-Gaussian Noise: Regression vs KCI
**Scenario:**  
- Z ~ Uniform(-3,3)  
- X = sin(Z) + Laplace noise  
- Y = cos(Z) + Laplace noise  
- X ⟂ Y | Z, but relation to Z is nonlinear and noise is non-Gaussian.

**Code:**
```python
import numpy as np
from scipy.stats import pearsonr, laplace
from sklearn.linear_model import LinearRegression
from pycitest import KCI

# Simulate
np.random.seed(42)
n = 400
Z = np.random.uniform(-3, 3, (n,1))
X = np.sin(Z) + 0.4 * laplace.rvs(size=(n,1))
Y = np.cos(Z) + 0.4 * laplace.rvs(size=(n,1))

# Partial correlation (linear)
lr_x = LinearRegression().fit(Z, X)
lr_y = LinearRegression().fit(Z, Y)
rx = X - lr_x.predict(Z)
ry = Y - lr_y.predict(Z)
r, p_lin = pearsonr(rx.ravel(), ry.ravel())
print("Partial corr:", r, "p:", p_lin)

# KCI
kci = KCI()
stat, p_kci = kci.test(X, Y, Z)
print("KCI stat:", stat, "p:", p_kci)
```
**Expected:** Partial corr p small (false dependence), KCI p large (correct independence).

---
