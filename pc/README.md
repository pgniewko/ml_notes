# PC Algorithm & Conditional Independence Tests — Comprehensive Tutorial

This tutorial is a self-contained guide to the Peter–Clark (PC) algorithm, the fundamental causal concepts it relies on, and practical conditional independence (CI) tests for both discrete and continuous data. It ends with an comparison on a nonlinear, non-Gaussian scenario.

---

## 0) What the PC algorithm does

Given observational data, PC learns as much of the causal graph as is identifiable by testing conditional independencies, pruning edges, then orienting some of the remaining ones to produce a CPDAG (a partially directed graph that represents an entire Markov equivalence class of DAGs).

---

## 1) Fundamental concepts

### 1.1 Directed Acyclic Graph (DAG) & d-separation
A DAG encodes causal relationships with directed edges and no directed cycles. d-separation is a graphical rule that, given a DAG, implies which conditional independencies hold in the induced probability distribution.

### 1.2 Causal Markov Condition
In the true causal DAG, each variable is independent of its non-descendants given its parents.

Formal, for node X with parent set Pa(X):
X _||_ NonDesc(X) | Pa(X)

Example. In A -> B -> C, once you condition on B, variable C is independent of A: C _||_ A | B.

Why it matters. It lets us infer statistical CI (what we can test) from graph structure (what we want). PC inverts this: it uses empirical CI to reason about the graph.

### 1.3 Faithfulness (a.k.a. Stability)
All and only the CI relations observed in the data arise from d-separations in the true DAG (no “coincidental” cancellations).

Violations. Exact parameter cancellations, deterministic relations, or selection effects can create CI patterns not explained by the graph. Then PC can mistakenly remove or keep edges or misorient v-structures.

Example (violation). In A -> B -> C, finely tuned parameters might make A and C look independent (even unconditionally). That contradicts what the DAG predicts; faithfulness fails.

### 1.4 Colliders (v-structures) & the collider orientation rule
A collider is a triple A -> B <- C (with A and C nonadjacent).

Unshielded triple: A - B - C with A and C not directly connected.

Collider orientation rule in PC. After edge pruning, for every unshielded triple A - B - C:
- If B is NOT in Sep(A,C) (i.e., A and C were found independent without conditioning on B), orient it as A -> B <- C (a collider).
- If B is in Sep(A,C), leave it unoriented (not a compelled collider).

Worked example. Suppose A and C are independent (unconditionally), but become dependent when conditioning on B (“explaining away”). Then B is a collider and PC will orient A -> B <- C.

### 1.5 Meek’s orientation rules (R1–R4)
After orienting colliders, Meek’s rules propagate directions without creating cycles or unsupported colliders.

- R1 (Propagation): If A -> B - C and A and C are nonadjacent, orient B - C as B -> C.
- R2 (Avoid new colliders): If A - B and there exists A -> C -> B, orient A - B as A -> B.
- R3 (Cycle avoidance via triangles): If A - B, A - C, with A -> C and C -> B, orient A - B as A -> B.
- R4 (Two distinct parents into one child): If A - B and there exist distinct Z1,Z2 such that A - Z1, Z1 -> B, A - Z2, Z2 -> B, and Z1 and Z2 are nonadjacent, orient A -> B.

Why they matter. They squeeze more directions from what CI tests alone identify, while respecting acyclicity and avoiding unsupported v-structures.

### 1.6 Markov equivalence & CPDAG (why observational data is not enough)
Two DAGs are Markov equivalent iff they have the same skeleton (adjacencies) and the same v-structures. Observational data (CI tests) cannot distinguish between DAGs within the same equivalence class.

- Equivalent example: A -> B -> C and A <- B -> C share the same skeleton A-B-C and have no colliders -> equivalent.
- Non-equivalent example: A -> B <- C has a collider at B; it is not equivalent to the previous two.

PC’s output is a CPDAG: a partially directed graph where directed edges are compelled (same direction in all DAGs in the class) and undirected edges are ambiguous across the class.

---

## 2) The PC algorithm step-by-step

1) Initialize a complete undirected graph on variables V. For each pair (X,Y), set the separating set Sep(X,Y)=∅.
2) Skeleton pruning by CI tests.
   - For ℓ = 0,1,2,...: for each adjacent pair (X,Y), test X _||_ Y | S for all subsets S ⊆ Adj(X)\{Y} with |S|=ℓ. If any such test accepts independence (p ≥ α), remove edge X—Y and record one such S as Sep(X,Y) (symmetrically).
   - Stop when no tests remain.
3) Orient colliders using the recorded separating sets (unshielded triples rule above).
4) Apply Meek’s rules R1–R4 iteratively until closure (no more orientations possible).
5) Return the CPDAG.

Why certain steps matter.
- Increasing |S| controls the combinatorics: try small conditioning sets first (often most informative and statistically stable).
- Recording Sep(X,Y) is critical to orient colliders correctly.
- Meek rules add more orientations without new CI tests.

Computational note. The skeleton phase is the bottleneck: worst-case roughly O(n^2 * 2^d) CI tests where d is max degree; orientation is polynomial (~O(n^3)). PC scales best on sparse graphs.

---

## 3) CI tests you’ll actually use

### 3.1 Discrete data

#### 3.1.1 G-test (likelihood-ratio test) for independence
Given observed counts O_ij and expected counts E_ij under independence:
G = 2 * sum_ij O_ij * log(O_ij / E_ij)  ~  chi-square(df)  (asymptotically),
with df=(r-1)(c-1) for an r x c table.

Python (plain and conditional):
```python
import pandas as pd
from scipy.stats import chi2_contingency, chi2

def g_test_discrete(X, Y):
    """Unconditional G-test using pd.crosstab and chi2_contingency(lambda_='log-likelihood')."""
    ct = pd.crosstab(X, Y)
    G, p, dof, _ = chi2_contingency(ct, lambda_="log-likelihood")
    return G, p, dof

def g_test_conditional(X, Y, Z):
    """Conditional G-test by stratifying on Z and summing G across strata; p via chi2 CDF."""
    df = pd.DataFrame({"X": X, "Y": Y, "Z": Z})
    G_total, df_total = 0.0, 0
    for _, grp in df.groupby("Z"):
        ct = pd.crosstab(grp["X"], grp["Y"])
        if ct.shape[0] > 1 and ct.shape[1] > 1:
            G, _, dof, _ = chi2_contingency(ct, lambda_="log-likelihood")
            G_total += G
            df_total += dof
    from scipy.stats import chi2
    p_value = 1 - chi2.cdf(G_total, df_total) if df_total > 0 else 1.0
    return G_total, p_value, df_total
```

#### 3.1.2 Pearson’s chi-square test (as a cross-check)
Same interface; replace lambda_='log-likelihood' with lambda_=None. G and chi-square are asymptotically equivalent; G often behaves slightly better when expected counts are small (still watch for cells with expected < 5).

```python
def chi2_test_conditional(X, Y, Z):
    df = pd.DataFrame({"X": X, "Y": Y, "Z": Z})
    chi2_total, df_total = 0.0, 0
    for _, grp in df.groupby("Z"):
        ct = pd.crosstab(grp["X"], grp["Y"])
        if ct.shape[0] > 1 and ct.shape[1] > 1:
            chi2_stat, _, dof, _ = chi2_contingency(ct, lambda_=None)
            chi2_total += chi2_stat
            df_total += dof
    from scipy.stats import chi2
    p_value = 1 - chi2.cdf(chi2_total, df_total) if df_total > 0 else 1.0
    return chi2_total, p_value, df_total
```

---

### 3.2 Continuous, approximately linear-Gaussian

#### Partial correlation / Fisher-Z (regress-out then test)
Residualize X and Y on Z via linear regression; test correlation of residuals.

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr, t

def partial_corr_test(X, Y, Z):
    """Test X independent of Y given Z under linear-Gaussian assumptions using residual correlation."""
    X = np.asarray(X); Y = np.asarray(Y); Z = np.asarray(Z)
    if Z.ndim == 1: Z = Z.reshape(-1, 1)
    rx = (X - LinearRegression().fit(Z, X).predict(Z)).ravel()
    ry = (Y - LinearRegression().fit(Z, Y).predict(Z)).ravel()
    r, _ = pearsonr(rx, ry)
    n = len(rx); dof = n - 2
    r = np.clip(r, -0.999999, 0.999999)
    t_stat = r * np.sqrt(dof / (1 - r**2))
    p = 2 * (1 - t.cdf(abs(t_stat), dof))
    return r, t_stat, p, dof
```

Caveats. Sensitive to nonlinearity, heteroskedasticity, and heavy tails/outliers (e.g., Laplace/Cauchy).

---

### 3.3 Continuous, nonlinear and/or non-Gaussian

#### 3.3.1 KCI — Kernel-based Conditional Independence
Nonparametric test using kernel embeddings (RKHS). Robust to nonlinear relations and non-Gaussian noise.

```python
# pip install pycitest
import numpy as np
try:
    from pycitest import KCI
except Exception:
    KCI = None

def kci_test(X, Y, Z, **kwargs):
    """Run KCI; returns (statistic, p_value)."""
    if KCI is None:
        raise ImportError("Install pycitest to run KCI: pip install pycitest")
    kci = KCI(**kwargs)  # kernels default to RBF; can pass sigmaX, sigmaY, sigmaZ
    stat, p = kci.test(np.asarray(X), np.asarray(Y), np.asarray(Z))
    return stat, p
```

#### 3.3.2 Conditional Mutual Information (CMI) — kNN estimator
Estimates I(X;Y|Z). A permutation test yields a p-value.

```python
# pip install npeet
import numpy as np
try:
    from npeet.entropy_estimators import cmi
except Exception:
    cmi = None

def cmi_perm_test(X, Y, Z, n_perm=200, seed=42):
    """Right-tailed permutation test on kNN CMI: higher CMI => more dependence."""
    if cmi is None:
        raise ImportError("Install NPEET to compute kNN CMI: pip install npeet")
    rng = np.random.default_rng(seed)
    X = np.asarray(X).ravel(); Y = np.asarray(Y).ravel(); Z = np.asarray(Z)
    obs = float(cmi(X, Y, Z))
    ge = 0
    for _ in range(n_perm):
        Yp = rng.permutation(Y)
        val = float(cmi(X, Yp, Z))
        if val >= obs: ge += 1
    p = (ge + 1) / (n_perm + 1)
    return obs, p
```

---

## 4) Orientation rules — quick reference

| Rule | Main use | Pattern / preconditions | Orientation logic | Why it matters |
|---|---|---|---|---|
| Collider | Identify v-structures | A - B - C, A and C nonadjacent, B not in Sep(A,C) | A -> B <- C | First compelled directions |
| R1 | Propagation | A -> B - C, A and C nonadjacent | B -> C | Extends directions without CI tests |
| R2 | Avoid new colliders | A - B and A -> C -> B | A -> B | Prevents unsupported v-structures |
| R3 | Cycle avoidance | A - B, A - C, A -> C, C -> B | A -> B | Keeps DAG acyclic |
| R4 | Two parents into B | A - B with distinct Z1 -> B, Z2 -> B, A - Z1, A - Z2, Z1 not adjacent Z2 | A -> B | Leverages converging evidence |

Markov equivalence reminder. Even after all rules, some edges stay undirected in the CPDAG because observational data cannot fix their direction (multiple DAGs fit the same CI pattern).

---

## 5) Methods at a glance — what to use when

| X,Y type | Z type | Relation | Noise | Recommended CI test | Python function |
|---|---|---|---|---|---|
| Discrete | Discrete | Any | Any | G-test (or chi-square) | g_test_discrete, g_test_conditional, chi2_test_conditional |
| Continuous | Any (small) | Linear | Gaussian-ish | Partial corr / Fisher-Z | partial_corr_test |
| Continuous | Any | Nonlinear | Non-Gaussian | KCI | kci_test |
| Continuous | Any | Nonlinear | Non-Gaussian | kNN-CMI + permutation | cmi_perm_test |
| Mixed | Mixed | Any | Any | Discretize / copula-based / CRT | (not included here) |

---

## 6) Minimal simulation helpers (for quick demos)

```python
import numpy as np
from scipy.stats import laplace

def simulate_discrete_xy_given_z(n=500, kx=3, ky=3, kz=2, seed=0):
    rng = np.random.default_rng(seed)
    Z = rng.integers(0, kz, size=n)
    P_x_given_z = rng.dirichlet(np.ones(kx), size=kz)
    P_y_given_z = rng.dirichlet(np.ones(ky), size=kz)
    X = np.array([rng.choice(kx, p=P_x_given_z[z]) for z in Z])
    Y = np.array([rng.choice(ky, p=P_y_given_z[z]) for z in Z])
    return X, Y, Z

def simulate_nonlinear_continuous(n=600, noise=0.35, seed=0):
    rng = np.random.default_rng(seed)
    Z = rng.uniform(-3, 3, size=(n,1))
    X = np.sin(Z) + noise * laplace.rvs(size=(n,1), random_state=seed)
    Y = np.cos(Z) + noise * laplace.rvs(size=(n,1), random_state=seed+1)
    return X, Y, Z
```

---

## 7) End-to-end demos

### 7.1 Discrete conditional independence (X _||_ Y | Z)
```python
X, Y, Z = simulate_discrete_xy_given_z(n=1000, kx=3, ky=4, kz=3, seed=1)
G, p, df = g_test_conditional(X, Y, Z)
print(f"[Discrete] Conditional G-test: G={G:.2f}, df={df}, p={p:.3g}")
```

### 7.2 Continuous — linear vs nonlinear tests on a nonlinear, non-Gaussian scenario
```python
Xc, Yc, Zc = simulate_nonlinear_continuous(n=800, noise=0.4, seed=2)

# Standard linear-Gaussian method (often fails here)
r, t_stat, p_lin, dof = partial_corr_test(Xc, Yc, Zc)
print(f"[Partial corr] r={r:.3f}, t={t_stat:.2f}, p={p_lin:.3g}")

# KCI (intended to handle nonlinearity and non-Gaussian noise)
try:
    stat_kci, p_kci = kci_test(Xc, Yc, Zc)
    print(f"[KCI] stat={stat_kci:.4f}, p={p_kci:.3g}")
except ImportError as e:
    print("KCI not available; install pycitest to run this: pip install pycitest")

# kNN CMI with permutation (alternative nonparametric)
try:
    obs_cmi, p_cmi = cmi_perm_test(Xc, Yc, Zc, n_perm=300, seed=3)
    print(f"[CMI] value={obs_cmi:.4f}, perm p={p_cmi:.3g}")
except ImportError:
    print("NPEET not available; install with: pip install npeet")
```

Expected interpretation.
- partial_corr_test often yields a small p-value (false positive), because residualizing with a linear model can’t remove the sin/cos structure.
- kci_test and/or cmi_perm_test typically return large p-values (correctly failing to reject X _||_ Y | Z).

---

## 8) Practical notes & pitfalls

- Choice of CI test controls PC’s behavior. Use discrete tests for discrete variables; for continuous data, prefer linear tests only when linear-Gaussian assumptions are plausible.
- Multiple testing & alpha. PC runs many CI tests; consider false discovery control or stability selection in practice.
- Sample size matters. Large conditioning sets can overfit; small cells in contingency tables can invalidate asymptotics.
- Hidden confounding. Standard PC assumes causal sufficiency. If that’s dubious, consider FCI variants.

---

*End of tutorial.*
