
# PC Algorithm & Conditional Independence Tests — Tutorial README

This README is for a tutorial on the **Peter–Clark (PC) algorithm** for causal discovery. It explains all crucial concepts (Causal Markov, Faithfulness, v-structures, Meek’s rules, Markov equivalence), details **conditional independence tests** for discrete and continuous data (linear and nonlinear), and compares methods under challenging scenarios (nonlinear dependencies, non-Gaussian noise).

---

## 1. Key Concepts

### 1.1 Causal Markov Condition
**Definition:** In a causal DAG, each variable is independent of its non-descendants given its direct parents.

**Formal:** For variable X:  
\[ X \perp NonDescendants(X) \mid Parents(X) \]

**Example:** In `A → B → C`, C is independent of A given B.

**Why it matters:** Allows translating d-separation in the graph to conditional independence in the data, which is the backbone of the PC algorithm’s edge removal step.

---

### 1.2 Faithfulness Condition
**Definition:** The only conditional independencies in the data are those implied by the DAG's d-separation.

**Example (Violation):** `A → B → C` but the parameters cause the influence of A on C to cancel out, making A and C independent despite the DAG implying dependence.

**Why it matters:** Without faithfulness, CI tests may mislead PC, producing incorrect graph structure.

---

### 1.3 Colliders (V-structures)
**Definition:** A node B is a collider if two arrows converge into it: `A → B ← C`.

**Identification in PC:** In an unshielded triple `A - B - C` (A and C nonadjacent), if B is not in the separating set for A and C, orient as `A → B ← C`.

**Example:** Suppose A and C are marginally independent, but conditioning on B makes them dependent — B is a collider.

**Why it matters:** Colliders are the first edges oriented in PC; correct collider detection is critical for downstream orientation rules.

---

### 1.4 Meek’s Rules (Comprehensive)
Applied iteratively after collider orientation to orient additional edges while preserving DAG properties.

- **R1 (Propagation):** If `A → B - C` and A, C are nonadjacent, orient `B → C`.
- **R2 (Avoid new colliders):** If `A - B` and `A → C → B`, orient `A → B`.
- **R3 (Cycle avoidance):** If `A - B`, `A - C`, `C → B`, `A → C`, orient `A → B`.
- **R4 (Two parents into one):** If `A - B` and there exist Z1, Z2 such that `A - Z1`, `Z1 → B`, `A - Z2`, `Z2 → B`, and Z1, Z2 are nonadjacent, orient `A → B`.
- **R5 (Extended propagation):** Avoid orientations that would introduce unsupported v-structures.

**Why it matters:** Increases the number of directed edges without new CI tests; critical for reducing Markov equivalence class ambiguity.

---

### 1.5 Markov Equivalence & CPDAG
**Definition:** Two DAGs are Markov equivalent if they have the same skeleton (adjacencies) and same set of v-structures.

**Implication:** Observational data alone cannot distinguish between DAGs in the same class.

**Example:** `A → B → C` and `A ← B → C` are Markov equivalent — same skeleton, no colliders.

**PC Output:** A CPDAG (Completed Partially Directed Acyclic Graph) encoding all DAGs in the equivalence class — directed edges are common to all DAGs, undirected edges are ambiguous.

---

## 2. Summary Table: Colliders & Meek Rules

| Rule Type | Main Use | Pattern | Orientation Logic |
|-----------|----------|---------|-------------------|
| Collider  | Identify v-structures | `A - B - C` (A,C nonadjacent, B ∉ Sep(A,C)) | `A → B ← C` |
| R1        | Propagation | `A → B - C`, A,C nonadjacent | `B → C` |
| R2        | Avoid new colliders | `A - B` with `A → C → B` | `A → B` |
| R3        | Cycle avoidance | `A - B`, `A - C`, `C → B`, `A → C` | `A → B` |
| R4        | Two parents | See description | `A → B` |
| R5        | Extended propagation | Various | Avoid unsupported v-structures |

---

## 3. Why Certain PC Steps Matter
- **Skeleton learning:** Removes false positives before orientation.
- **Collider orientation:** Introduces first causal directions.
- **Meek’s rules:** Expands orientations logically.
- **Faithfulness assumption:** Links CI patterns to graph topology.

---

## 4. Conditional Independence Tests

### 4.1 G-Test for Discrete Data
**Definition:** Likelihood-ratio test for independence.

**Formula:**  
\[ G = 2 \sum_{i,j} O_{ij} \log\left( \frac{O_{ij}}{E_{ij}} \right) \]  
Where O = observed counts, E = expected counts under independence.

**Python Example:**
```python
import pandas as pd
from scipy.stats import chi2_contingency, chi2

def g_test_discrete(X, Y):
    ct = pd.crosstab(X, Y)
    G, p, dof, exp = chi2_contingency(ct, lambda_="log-likelihood")
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

---

### 4.2 Chi-Squared Test for Discrete Data
Similar to G-test but uses Pearson’s chi-squared statistic.

```python
def chi2_test_conditional(X, Y, Z):
    df = pd.DataFrame({"X": X, "Y": Y, "Z": Z})
    chi2_total, df_total = 0, 0
    for _, group in df.groupby("Z"):
        ct = pd.crosstab(group.X, group.Y)
        if ct.shape[0] > 1 and ct.shape[1] > 1:
            chi2_stat, _, dof, _ = chi2_contingency(ct, lambda_=None)
            chi2_total += chi2_stat
            df_total += dof
    p_value = 1 - chi2.cdf(chi2_total, df_total)
    return chi2_total, p_value, df_total
```

---

### 4.3 Partial Correlation (Continuous, Linear Gaussian)
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

### 4.4 Conditional Mutual Information (CMI)
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

### 4.5 KCI (Kernel-based Conditional Independence)
Uses kernel embeddings to test CI robustly to nonlinearities and non-Gaussianity.

---

## 5. Nonlinear & Non-Gaussian Scenario Comparison
**Scenario:** Z ~ Uniform(-3,3); X = sin(Z) + Laplace noise; Y = cos(Z) + Laplace noise; X ⟂ Y | Z.

**Code:**
```python
from scipy.stats import laplace
np.random.seed(42)
n = 400
Z = np.random.uniform(-3, 3, (n,1))
X = np.sin(Z) + 0.4 * laplace.rvs(size=(n,1))
Y = np.cos(Z) + 0.4 * laplace.rvs(size=(n,1))

# Partial correlation
lr_x = LinearRegression().fit(Z, X)
lr_y = LinearRegression().fit(Z, Y)
rx = X - lr_x.predict(Z)
ry = Y - lr_y.predict(Z)
r, p_lin = pearsonr(rx.ravel(), ry.ravel())
print("Partial corr:", r, "p:", p_lin)

# KCI
from pycitest import KCI
kci = KCI()
stat, p_kci = kci.test(X, Y, Z)
print("KCI stat:", stat, "p:", p_kci)
```
**Expected:** Partial correlation p small (false positive); KCI p large (correct independence).

---
