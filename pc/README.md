# Peter–Clark Algorithm Tutorial

A concise, practical guide to the PC algorithm for learning causal structure from observational data.

---

## Table of Contents
- [0. Notation](#0-notation)
- [1. Assumptions](#1-assumptions)
  - [1.1 Causal Markov Condition](#11-causal-markov-condition)
  - [1.2 Faithfulness](#12-faithfulness)
- [2. Markov Equivalence & Completed Partially Directed Acyclic Graph](#2-markov-equivalence--completed-partially-directed-acyclic-graph)
- [3. The PC Algorithm: High-Level Overview](#3-the-pc-algorithm-high-level-overview)
  - [3.1 Edge Deletion - Detailed Conditioning Strategy](#31-edge-deletion---detailed-conditioning-strategy) 
- [4. Orientation Rules](#4-orientation-rules)
  - [4.1 Collider (v-structure) Orientation](#41-collider-v-structure-orientation)
  - [4.2 Meek’s Rules](#42-meeks-rules)
- [5. Conditional Independence (CI) Tests](#5-conditional-independence-ci-tests)
  - [5.1 Discrete Data: Chi-squared and G-test](#51-discrete-data-chi-squared-and-g-test)
  - [5.2 Continuous Data: Partial Correlation](#52-continuous-data-partial-correlation)
  - [5.3 Continuous/General: KCI and CMI](#53-continuousgeneral-kci-and-cmi)
- [6. Illustrative Nonlinear Example: Regression vs KCI](#6-illustrative-nonlinear-example-regression-vs-kci)
- [7. Computational Complexity](#7-computational-complexity)
- [8. Summary Tables](#8-summary-tables)

---
## 0. Notation
X ⫫ Y | Z denotes *“X is conditionally independent of Y given Z”*.  
Formally, for all values of $Z$:

$$
P(X, Y \mid Z) = P(X \mid Z) P(Y \mid Z)
$$

This means that once $Z$ is known, learning $Y$ provides no additional information about $X$ (and vice versa).

## 1. Assumptions

The PC algorithm assumes the true causal structure is a Directed Acyclic Graph (DAG) and that the data-generating process satisfies the **Causal Markov** and **Faithfulness** conditions.

### 1.1 Causal Markov Condition

**Definition.** In a causal DAG, each variable is independent of its non-descendants given its direct parents.

**Intuition.** Each node is generated from its parents and independent noise; once parents are known, additional non-descendant information doesn’t help.

**Example.** For $A \to B \to C$, we have C ⫫ A | B.

### 1.2 Faithfulness

**Definition.** All and only the conditional independencies in the data are those implied by the DAG structure (no “coincidental” cancellations).

**Why it matters.** Violations (e.g., parameter cancellations) can mislead CI tests and, hence, PC’s edge deletions.

**Example.** If the true DAG is $A \to B \to C$ but, by chance, $A$ and $C$ appear independent in the sample, the process is unfaithful.

---

## 2. Markov Equivalence & Completed Partially Directed Acyclic Graph

- **Limit of observational data.** Many DAGs entail the same set of CI relations; they form a **Markov equivalence class**.
- **Two DAGs are Markov equivalent** _iff_ they share:
  1) the same **skeleton** (adjacencies) and
  2) the same set of **v-structures** (colliders $X \to Y \leftarrow Z$, with $X$ and $Z$ non-adjacent).
- **Output of PC.** A **CPDAG** (Completed Partially Directed Acyclic Graph) that encodes the entire equivalence class:
  - **Directed edges** are oriented the same way in all DAGs of the class;
  - **Undirected edges** are ambiguous (could be either direction).

**Example.** $A \to B \to C$ and $A \leftarrow B \to C$ share skeleton $A - B - C$ and have no colliders → **Markov equivalent**. But $A \to B \leftarrow C$ has a collider at $B$ and is **not** equivalent to the others.

---

## 3. The PC Algorithm: High-Level Overview

1. **Start with a complete undirected graph.** Every pair of variables is connected.
2. **Edge deletion via CI tests.** Iteratively test (conditional) independence for pairs, conditioning on increasingly large subsets. If X ⫫ Y | Z, remove the edge $X - Y$.
3. **Orient edges using rules.** First orient colliders; then apply **Meek’s rules** to propagate orientations to a maximally informative **CPDAG**.

### 3.1 Edge Deletion - Detailed Conditioning Strategy  
The skeleton pruning phase removes edges by testing conditional independence for $(X, Y)$ with increasingly large conditioning sets $Z$:

- **|Z| = 0 (No conditioning):** Test unconditional independence. If X ⫫ Y, remove the edge.
- **|Z| = 1 (Single-variable):** Test X ⫫ Y | Z for each single neighbor. Removes edges explained away by one variable.
- **|Z| = 2 (Pairs):** Test with all neighbor pairs as conditioning sets.
- **|Z| ≥ 3:** Increase $|Z|$ until no more subsets remain or max size reached.

**Notes:**
- Choose $Z$ from the adjacency set of the tested node (excluding the other variable).
- If *any* $Z$ makes $X$ and $Y$ independent (p ≥ α), delete the edge and store $Sep(X,Y)$.
- Small conditioning sets are statistically stronger and computationally cheaper; large sets risk overfitting.


---

## 4. Orientation Rules

Notation: $\to$ := directed edge; $-$ := undirected; non-adjacency stated explicitly.

### 4.1 Collider (v-structure) Orientation

A **collider** has the form $A \to B \leftarrow C$. PC orients colliders as follows:

- After skeleton discovery, suppose $A - B - C$ and $A$ is **not** adjacent to $C$.
- If $B$ was **not** included in any conditioning set that rendered A ⫫ C | S, **orient as a collider**: $A \to B \leftarrow C$.
- If $B$ was in such a conditioning set, **do not** orient as a collider.

This is PC’s primary source of initial edge directions.

### 4.2 Meek’s Rules

- **Rule 1 (Propagation).** If $A \to B - C$ and $A$ is not adjacent to $C$, orient $B - C$ as $B \to C$.
- **Rule 2 (Avoid new colliders).** If $A - B$, $A \to C$, $C \to B$, and $A$ not adjacent to $B$, orient $A - B$ as $A \to B$.
- **Rule 3 (Cycle avoidance).** If $A - B$, $A - C$, $A \to D$, $C \to D$, and $B \to D$, orient $A - B$ as $A \to B$.
- **Rule 4 (More propagation).** If $A - B$, $A \to C$, $B \to C$, orient $A - B$ as $A \to B$.

These rules propagate directions while respecting acyclicity and avoiding unsupported colliders.

---

## 5. Conditional Independence (CI) Tests

### 5.1 Discrete Data: Chi-squared and G-test
**Idea:** For discrete variables, use the **G-test** (likelihood ratio test) or **chi-squared test** on a contingency table of observed vs. expected counts under the null hypothesis of conditional independence.  

**G-test (likelihood ratio) statistic**

$$
G = 2 \sum_{i=1}^{r} \sum_{j=1}^{c} O_{ij} \ln\left(\frac{O_{ij}}{E_{ij}}\right)
$$

with degrees of freedom $\mathrm{dof} = (r-1)(c-1)$, 
where $O$ and $E$ are observed and expected counts.

**Why this works:**  
- Conditional independence X ⫫ Y | Z means that, within each level (or stratum) of $Z$, the joint distribution of $X$ and $Y$ factorizes:
  
$$
  P(X, Y \mid Z) = P(X \mid Z) \cdot P(Y \mid Z)
$$

- This factorization implies that the **expected cell counts** in the contingency table for $X$ and $Y$ given $Z$ can be computed as the product of their marginal probabilities within that stratum.  
- **G-test:** compares observed counts $O_{ij}$ to expected counts $E_{ij}$ using G statistic.
  
  Large values of $G$ indicate deviation from independence.
- **Chi-squared test:** uses
  
$$
  \chi^2 = \sum_{i,j} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}
$$
  
  to measure the same deviation but via squared differences.
  
- When conditioning on $Z$, the test is repeated within each stratum defined by $Z$, and the statistics are summed across strata.  
- If the summed statistic is **small** (and p-value large), the data is consistent with the null hypothesis that $X$ and $Y$ are independent given $Z$. If it’s **large** (small p-value), it suggests conditional dependence.


**Python (G-test via `scipy.stats`):**
```python
import numpy as np
from scipy.stats import chi2_contingency

# Example 2x2 contingency table (A vs B)
obs = np.array([[10, 20],
                [20, 40]])

# Use lambda_="log-likelihood" to obtain the G-test
g_stat, p, dof, expected = chi2_contingency(obs, lambda_="log-likelihood")
print(f"G statistic: {g_stat:.3f}, p-value: {p:.3g}, dof={dof}")
```

**Stratified (conditional) version.** Repeat the test within each stratum of $Z$ and combine statistics (e.g., sum $G$ across strata).

**Utility function for CI in discrete data:**
```python
import pandas as pd
from scipy.stats import chi2_contingency

def ci_test_discrete(df, x, y, z=None):
    """
    Chi-squared/G-test based CI test for discrete variables.
    df : pandas.DataFrame of discrete variables
    x, y : column names
    z : list of conditioning column names (optional)
    Returns: p-value
    """
    z = z or []
    if not z:
        ct = pd.crosstab(df[x], df[y])
        _, p, _, _ = chi2_contingency(ct)
        return p

    p_vals = []
    for _, group in df.groupby(z):
        if group[x].nunique() > 1 and group[y].nunique() > 1:
            ct = pd.crosstab(group[x], group[y])
            _, p, _, _ = chi2_contingency(ct)
            p_vals.append(p)

    return min(p_vals) if p_vals else 1.0
```

### 5.2 Continuous Data: Partial Correlation

**Idea.** Regress $X$ and $Y$ on $Z$, then test correlation between residuals. Suitable under linear-Gaussian assumptions.

**Why this works:**  
If $X$ and $Y$ are conditionally independent given $Z$, then any statistical relationship between them should be fully explained by $Z$.  
By regressing each variable on $Z$, we remove the linear component of their dependence on $Z$, leaving residuals $r_X$ and $r_Y$ that represent the variation unexplained by $Z$.  
- If X ⫫ Y | Z holds under the linear-Gaussian model, $r_X$ and $r_Y$ will be *uncorrelated* and, in the Gaussian case, uncorrelatedness implies independence.  
- If the residuals are still correlated, it means there is remaining association between $X$ and $Y$ after controlling for $Z$, which violates conditional independence.

```python
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

def partial_correlation(x, y, z):
    """
    x, y : arrays of shape (n,) or (n, 1)
    z    : array of shape (n,) or (n, k) (conditioning set)
    Returns: (corr, p_value)
    """
    z = z.reshape(-1, 1) if z.ndim == 1 else z

    # Residualize x on z
    lr_x = LinearRegression().fit(z, x)
    x_res = x - lr_x.predict(z)

    # Residualize y on z
    lr_y = LinearRegression().fit(z, y)
    y_res = y - lr_y.predict(z)

    # Correlate residuals
    corr, p_value = pearsonr(x_res.ravel(), y_res.ravel())
    return corr, p_value

# Example
np.random.seed(0)
n = 200
z = np.random.normal(0, 1, n)
x = 2*z + np.random.normal(0, 1, n)
y = -3*z + np.random.normal(0, 1, n)

corr, p = partial_correlation(x, y, z)
print(f"Partial correlation: {corr:.3f}, p-value: {p:.3g}")
```

### 5.3 Continuous/General: KCI and CMI

- **KCI (Kernel-based CI).** Nonparametric, detects nonlinear/non-Gaussian dependencies.
- **CMI (Conditional Mutual Information).** Information-theoretic; often estimated via $k$-NN for continuous variables.

**KCI (requires `pycitest`):**
```python
# pip install pycitest
import numpy as np
from pycitest import KCI

np.random.seed(42)
n = 200
Z = np.random.randn(n, 1)
X = Z + 0.1*np.random.randn(n, 1)
Y = Z + 0.1*np.random.randn(n, 1)

kci = KCI()
stat, p_value = kci.test(X, Y, Z)
print(f"KCI test statistic: {stat:.3f}, p-value: {p_value:.3g}")
```

**CMI (requires `npeet`):**
```python
# pip install npeet
import numpy as np
from npeet.entropy_estimators import cmi

np.random.seed(42)
n = 200
Z = np.random.randn(n)
X = Z + 0.1*np.random.randn(n)
Y = Z + 0.1*np.random.randn(n)

cmi_value = cmi(X, Y, Z)
print(f"Conditional mutual information: {cmi_value:.3f}")
```

---

## 6. Illustrative Nonlinear Example: Regression vs KCI

**Setup.** 
- Latent $Z$
- $X=\sin(Z)+$ Laplace noise
- $Y=\cos(Z)+$ Laplace noise
- Ground truth: X ⫫ Y | Z.

```python
# pip install pycitest
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr, laplace
from pycitest import KCI

np.random.seed(42)
n = 400
Z = np.random.uniform(-3, 3, (n, 1))
X = np.sin(Z) + 0.4 * laplace.rvs(size=(n, 1))
Y = np.cos(Z) + 0.4 * laplace.rvs(size=(n, 1))

# Standard approach: regress out Z, then correlate residuals
reg_x = LinearRegression().fit(Z, X)
reg_y = LinearRegression().fit(Z, Y)
X_resid = X - reg_x.predict(Z)
Y_resid = Y - reg_y.predict(Z)
corr, pval = pearsonr(X_resid.ravel(), Y_resid.ravel())
print(f"Partial correlation (linear regression): {corr:.3f}, p-value: {pval:.3g}")

# KCI
kci = KCI()
stat, kci_pval = kci.test(X, Y, Z)
print(f"KCI statistic: {stat:.3f}, p-value: {kci_pval:.3g}")
```

**Interpretation.** Linear regression removes only linear effects and assumes Gaussian noise; it often reports spurious dependence here (false positive). KCI handles nonlinearities/non-Gaussian noise and tends to yield a large $p$-value (correctly failing to reject independence).

---

## 7. Computational Complexity

- **Skeleton search (edge deletion).** For each pair $(X,Y)$, test CI over subsets of adjacent variables up to size $d$ (max degree). Worst-case number of tests is exponential in $d$.

  $$\text{Worst case: }\ \mathcal{O}\left(n^{2}2^{d}\right)$$

- **Edge orientation.** Application of orientation rules is typically polynomial, about $\mathcal{O}(n^{3})$.

**Implications.** PC scales well on sparse graphs (small $d$), but can be intractable for dense graphs or hubs. Practical variants (e.g., **PC-stable**, **FastPC**) reduce sensitivity to ordering and/or improve efficiency.

---

## 8. Summary Tables

### 8.1 Core Concepts
| Concept | Meaning |
|---|---|
| Conditional Independence | X ⫫ Y | Z: $X$ and $Y$ independent given $Z$ |
| Causal Markov | Each variable is independent of non-descendants given parents |
| Faithfulness | All and only independencies in data arise from the DAG structure |

### 8.2 Orientation Cheatsheet
| Concept | Main Use | Pattern | Orientation Logic |
|---|---|---|---|
| Collider | Orient v-structures | $A - B - C$, with $A$ and $C$ non-adjacent | If $B$ **not** in conditioning set that made A ⫫ C | S, orient $A \to B \leftarrow C$ |
| Meek’s Rules | Propagate & avoid ambiguity | Mixed directed/undirected patterns | Orient to avoid cycles, avoid new unsupported colliders, and propagate clear directions |

### 8.3 Markov Equivalence & CPDAG
| Property | Same in Equivalence Class? |
|---|---|
| Skeleton (adjacencies) | Yes |
| Colliders (v-structures) | Yes |
| Other edge directions | No (can differ) |

### 8.4 CI Tests at a Glance
| Test | Use Case | Key Library | Handles Nonlinear? | Conditional? |
|---|---|---|---|---|
| G-test / $\chi^2$ | Discrete | `scipy.stats` | No | Yes (by stratification) |
| Partial correlation | Continuous (linear-Gaussian) | `scipy`, `sklearn` | No | Yes |
| KCI | Continuous/general | `pycitest` | Yes | Yes |
| CMI | Continuous/general | `npeet` | Yes | Yes |

---
