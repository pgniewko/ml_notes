# PC Algorithm & Conditional Independence Tests — Tutorial README

This README is for a tutorial on the **Peter–Clark (PC) algorithm** for causal discovery. It introduces key concepts like **Faithfulness**, **Causal Markov Condition**, **collider identification**, and **Meek rules**, then focuses on **conditional independence (CI) testing** for discrete and continuous variables, with runnable Python examples.

---

## Key Concepts

### Causal Markov Condition
In a DAG, each node is conditionally independent of its non-descendants given its parents. This lets us map graph separation to statistical independence.

### Faithfulness
All and only the conditional independencies in the data come from the DAG's structure (no coincidental cancellations). Without faithfulness, CI tests can mislead structure learning.

### Colliders
Pattern **A → B ← C** where A and C are not connected. Identified when A and C are independent given some set S, but **B ∉ S**.

### Meek Rules
Propagation rules for orienting undirected edges without introducing new colliders or cycles. Examples:
- Rule 1: If A → B – C and A and C are not adjacent, orient B – C as B → C.
- Rule 2: Avoid new colliders by orienting edges consistent with existing orientations.

### Markov Equivalence
Multiple DAGs can have the same skeleton and collider set. PC outputs a **CPDAG** representing this equivalence class.

---

## PC Algorithm Steps
1. **Start:** Fully connected undirected graph.
2. **Skeleton learning:** Remove edges via CI tests (starting with unconditional, then conditional on 1 neighbor, etc.).
3. **Collider orientation:** Use separating sets to orient A → B ← C.
4. **Meek rules:** Propagate orientations until no further changes.

---

## Conditional Independence Tests

### Discrete Variables — G-Test (Likelihood Ratio)
```python
import pandas as pd
from scipy.stats import chi2_contingency, chi2

def conditional_g_test(X, Y, Z):
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

### Continuous Variables — Partial Regression (Linear Gaussian)
```python
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr, t
import numpy as np

def partial_corr_test(X, Y, Z):
    if Z.ndim == 1: Z = Z.reshape(-1, 1)
    lr_x = LinearRegression().fit(Z, X)
    lr_y = LinearRegression().fit(Z, Y)
    rx = X - lr_x.predict(Z)
    ry = Y - lr_y.predict(Z)
    r, _ = pearsonr(rx.ravel(), ry.ravel())
    n = len(rx)
    dof = n - 2
    t_stat = r * np.sqrt(dof / (1 - r**2))
    p_value = 2 * (1 - t.cdf(abs(t_stat), dof))
    return r, p_value
```

### Continuous Variables — KCI (Nonlinear)
Uses kernel regression to remove Z effects, then tests cross-covariance in RKHS. See `pycitest.KCI` for implementation.

---

## Method Selection Table

| Data Type   | Relationship Assumptions      | Recommended Method     |
|-------------|--------------------------------|------------------------|
| Discrete    | Any                            | G-test / Conditional G |
| Continuous  | Linear, Gaussian               | Partial correlation    |
| Continuous  | Nonlinear, Non-Gaussian        | KCI, kNN-based CMI     |

---

## Methods at a Glance: What to Use When

| Data type of X, Y | Conditioning set Z | Relationship shape | Noise | Recommended CI test | Notes |
|---|---|---|---|---|---|
| Discrete | Discrete | Any | Any | G-test (likelihood-ratio) or chi-squared (χ²); conditional by stratifying on Z | Sum statistics/df across Z strata with adequate expected counts |
| Continuous | None / small Z, approx. linear | Gaussian-ish | Partial correlation / Fisher Z | Fast, classical; may fail under nonlinearity or heavy tails |
| Continuous | Nonlinear | Non-Gaussian | KCI (kernel CI) | Kernel widths via median heuristic or CV; permutation/bootstrap p-values |
| Continuous | Nonlinear | Non-Gaussian | kNN CMI + permutation | Robust, but slower; parameter k matters |
| Mixed (discrete/continuous) | Any | Any | Discretize or use copula/KDE CI | Beware discretization bias; consider conditional randomization tests |

---

## PC Algorithm — Detailed Steps

Inputs: variables V, dataset D, CI test oracle (appropriate to data type), significance level alpha.

1. Initialize skeleton G as a complete undirected graph over V. Set separation sets Sep(X,Y)=∅.
2. Edge pruning by increasing conditioning size:
   - Let ell = 0,1,2,...
   - For each adjacent pair (X, Y) in G where both have at least ell neighbors (excluding each other), test X ⟂ Y | S for all S subset of Adj(X)\{Y} with |S| = ell. If independence holds at level alpha for any S, remove edge X—Y and record Sep(X,Y)=S (and symmetric).
   - Increase ell until no tests are possible.
3. Orient colliders (v-structures): For every unshielded triple X—Z—Y with X and Y nonadjacent:
   - If Z not in Sep(X,Y), orient as X → Z ← Y (collider). If Z is in Sep(X,Y), leave unoriented.
4. Apply Meek’s orientation rules repeatedly until no more orientations are possible (see below).
5. Return CPDAG representing the Markov equivalence class.

### Collider Identification (Unshielded Triples)
- Unshielded triple: X—Z—Y with X and Y not adjacent. Using the separating set from pruning, orient toward Z when Z was not used to separate X and Y. This creates a v-structure (collider) and is compelled in the CPDAG.

### Meek’s Rules (R1–R4)
Let "→" be directed, "—" undirected, and assume X and Z are nonadjacent unless stated.

- R1 (Orientation propagation): If X → Y — Z and X and Z are nonadjacent, then Y → Z.
- R2 (Avoid new colliders): If X — Y, and there exists X → Z → Y, then orient X → Y.
- R3 (Cycle avoidance via triangles): If X — Y, and there exists X — Z with Z → Y and X → Z, then orient X → Y.
- R4 (Two directed paths into a node): If X — Y and there exist distinct nodes Z1, Z2 such that X — Z1, Z1 → Y, X — Z2, Z2 → Y, and Z1 and Z2 are nonadjacent, then X → Y.

Apply R1–R4 iteratively until closure; they ensure no directed cycles are created and no unsupported colliders are introduced.

---

## Assumptions: Causal Markov & Faithfulness (Expanded)

- Causal Markov Condition: In the true causal DAG, each variable is independent of its non-descendants given its parents. This licenses reading d-separation as conditional independence in the data: if a set S d-separates X and Y in the DAG, then X ⟂ Y | S in the distribution.

- Faithfulness (Stability): All and only the conditional independencies present in the data arise from d-separation in the true DAG (no measure-zero parameter cancellations). Implication for PC: detected (conditional) independencies map back to missing edges and collider orientations. Violations (e.g., exact coefficient cancellations, deterministic relations, selection bias) can cause PC to remove/keep the wrong edges or misorient v-structures.

---

## Discrete CI with chi-squared (in addition to the G-test)

```python
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, chi2

# Pearson chi-squared test for (un)conditional independence

def chi2_test_independence(table):
    """Pearson chi-squared (χ²) test on a 2D contingency table. Returns (chi2_stat, p, df, expected)."""
    chi2_stat, p, df, expected = chi2_contingency(table, lambda_=None)
    return chi2_stat, p, df, expected


def conditional_chi2_test(X, Y, Z):
    """Conditional chi-squared by stratifying on discrete Z and summing χ² across strata.
    Returns (chi2_total, p_value, df_total)."""
    df = pd.DataFrame({"X": X, "Y": Y, "Z": Z})
    chi2_total, df_total = 0.0, 0
    for _, grp in df.groupby("Z"):
        ct = pd.crosstab(grp["X"], grp["Y"])  # observed in stratum
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            continue
        chi2_stat, p, dof, exp = chi2_contingency(ct, lambda_=None)
        chi2_total += chi2_stat
        df_total += dof
    p_value = 1 - chi2.cdf(chi2_total, df_total) if df_total > 0 else 1.0
    return chi2_total, p_value, df_total

# Example usage (compare with G-test):
# X, Y, Z = simulate_discrete_xy_given_z(n=600, kx=3, ky=3, kz=2)
# chi2_stat, p_chi2, df_total = conditional_chi2_test(X, Y, Z)
# print(f"Conditional χ²: χ²={chi2_stat:.2f}, df={df_total}, p={p_chi2:.3g}")
```

When to prefer G vs chi-squared? They are asymptotically equivalent; G often behaves better with small expected counts. If any expected cell < 5, consider merging categories or using exact tests.

---

## Partial Regression Code — Discussion & Caveats

Recall our linear CI test for continuous data:

```python
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr, t
import numpy as np

def partial_corr_test(X, Y, Z):
    """Test X ⟂ Y | Z via linear regression residuals + t-test on residual correlation."""
    X = np.asarray(X); Y = np.asarray(Y); Z = np.asarray(Z)
    if Z.ndim == 1: Z = Z.reshape(-1, 1)
    # Residualize
    rx = (X - LinearRegression().fit(Z, X).predict(Z)).ravel()
    ry = (Y - LinearRegression().fit(Z, Y).predict(Z)).ravel()
    # Correlate residuals and test
    r, _ = pearsonr(rx, ry)
    n, dof = len(rx), len(rx) - 2
    r = np.clip(r, -0.999999, 0.999999)
    t_stat = r * np.sqrt(dof / (1 - r**2))
    p = 2 * (1 - t.cdf(abs(t_stat), dof))
    return r, t_stat, p, dof
```

Interpretation: small p ⇒ evidence against CI (dependence remains after regressing out Z). Assumptions: linearity and homoscedastic, approximately Gaussian residuals. Pitfalls: nonlinear relations (e.g., sin/cos), heteroskedasticity, outliers and heavy tails (Laplace/Cauchy). In such cases, prefer KCI or kNN CMI.

---
