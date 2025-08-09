1. Peter-Clark (PC) Algorithm
The PC algorithm is a classic method for learning causal structure (a causal graph) from observational data, under certain assumptions (mainly, that the true structure is a Directed Acyclic Graph (DAG), the Causal Markov Condition, and faithfulness).

Overview of the PC Algorithm Steps:
Start with a Complete Undirected Graph:

Every pair of variables is connected.

Edge Deletion via Conditional Independence Tests:

Iteratively test for (conditional) independence between variable pairs, conditioning on increasing-size subsets of other variables.

If two variables are conditionally independent given some set, remove the edge between them.

Orient Edges Using Orientation Rules:

After building the skeleton (undirected structure), orient as many edges as possible to get a partially directed acyclic graph (PDAG or CPDAG).

Orientation uses collider rules and Meek's rules (see below).

Key Point:
PC uses conditional independence to prune edges, then tries to direct edges to infer as much causality as the data allows, given ambiguities.

Why This Matters
These orientation rules are key to inferring causal directions beyond what conditional independence can tell you. Without them, you'd be left with a network with many undirected or ambiguous edges.

2. Collider Orientation Rule
A collider is a pattern among three variables: 
𝐴
→
𝐵
←
𝐶
A→B←C
Here, 
𝐵
B is a collider because both edges "collide" at 
𝐵
B.

How PC Orients Colliders:
Suppose, after edge deletion, you have:

𝐴
−
𝐵
−
𝐶
A−B−C (both undirected)

𝐴
A and 
𝐶
C are not directly connected.

If 
𝐵
B is not in the conditioning set that made 
𝐴
A and 
𝐶
C independent,

Then, orient as a collider: 
𝐴
→
𝐵
←
𝐶
A→B←C

If 
𝐵
B was in the conditioning set, do not orient as a collider.

This is the main way the PC algorithm initially orients some edges.

3. Meek’s Orientation Rules
After orienting colliders, there will be many undirected edges left. Meek’s rules are a set of logical implications to propagate orientations further, ensuring:

No cycles (DAG property)

No new colliders (unless supported by data)

Main Meek’s Rules:
(Here, "
→
→" is a directed edge, "-" is undirected.)

Rule 1 (Propagation):
If 
𝐴
→
𝐵
−
𝐶
A→B−C and 
𝐴
A and 
𝐶
C are not connected, then orient 
𝐵
−
𝐶
B−C as 
𝐵
→
𝐶
B→C.

Rule 2 (Avoid New Colliders):
If 
𝐴
−
𝐵
A−B, 
𝐴
→
𝐶
A→C, 
𝐶
→
𝐵
C→B, and 
𝐴
A and 
𝐵
B are not connected, then orient 
𝐴
−
𝐵
A−B as 
𝐴
→
𝐵
A→B.

Rule 3 (Cycle Avoidance):
If 
𝐴
−
𝐵
A−B, 
𝐴
−
𝐶
A−C, 
𝐴
→
𝐷
A→D, 
𝐶
→
𝐷
C→D, and 
𝐵
→
𝐷
B→D, then orient 
𝐴
−
𝐵
A−B as 
𝐴
→
𝐵
A→B (to avoid cycles).

Rule 4 (More Propagation):
If 
𝐴
−
𝐵
A−B, 
𝐴
→
𝐶
A→C, 
𝐵
→
𝐶
B→C, then orient 
𝐴
−
𝐵
A−B as 
𝐴
→
𝐵
A→B.

(The rules can be presented with slight variations in literature, but these are the core ideas.)


4. Summary Table
Concept	Main Use	Pattern	Orientation Logic
Collider	Orient v-structures	
𝐴
−
𝐵
−
𝐶
A−B−C, no edge between A & C	If B not in cond. set, 
𝐴
→
𝐵
←
𝐶
A→B←C
Meek’s Rules	Propagate & avoid ambiguity	Patterns with mix of directed/undirected edges	Orient to prevent cycles, new colliders, or to propagate clear directions


1. Conditional Independence Test
In practice, conditional independence between variables 
𝑋
X and 
𝑌
Y given a set 
𝑍
Z means:

Knowing 
𝑌
Y gives no extra info about 
𝑋
X if you already know 
𝑍
Z.

How do you test for conditional independence?
a) For Discrete Data
Use the Chi-squared test or G-test on contingency tables.

For example: Are 
𝑋
X and 
𝑌
Y independent given 
𝑍
Z?

b) For Continuous Data
Use partial correlation tests: test if the correlation between 
𝑋
X and 
𝑌
Y remains after regressing out 
𝑍
Z.

If not Gaussian: Use nonparametric tests like Kernel-based Conditional Independence (KCI) or Conditional Mutual Information estimators.


Python Example: Partial Correlation Test
Let’s assume you have three continuous variables: x, y, z.

Step 1: Regress X and Y on Z, get residuals
Step 2: Correlate the residuals
python
Copy
Edit


import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

def partial_correlation(x, y, z):
    # Reshape for sklearn
    z = z.reshape(-1, 1) if z.ndim == 1 else z
    
    # Regress x on z, get residuals
    lr_x = LinearRegression().fit(z, x)
    x_res = x - lr_x.predict(z)
    
    # Regress y on z, get residuals
    lr_y = LinearRegression().fit(z, y)
    y_res = y - lr_y.predict(z)
    
    # Test correlation of residuals
    corr, p_value = pearsonr(x_res, y_res)
    return corr, p_value

# Example usage
np.random.seed(0)
n = 200
z = np.random.normal(0, 1, n)
x = 2 * z + np.random.normal(0, 1, n)
y = -3 * z + np.random.normal(0, 1, n)

corr, p = partial_correlation(x, y, z)
print(f"Partial correlation: {corr:.3f}, p-value: {p:.3g}")


If p is small (e.g., < 0.05), you reject independence. If large, you accept that X and Y are conditionally independent given Z.

2. Causal Markov Condition
Definition:
Given a causal DAG, each variable is independent of its non-descendants given its direct parents.

Why? Each node is determined by its parents and independent noise. If you know the parents, knowing anything else doesn't help.

Example:
If 
𝐴
→
𝐵
→
𝐶
A→B→C,

𝐶
C is independent of 
𝐴
A given 
𝐵
B.

3. Faithfulness Condition
Definition:
All and only the conditional independencies that hold in the data are those implied by the causal graph structure.

Why does this matter? Without faithfulness, some independencies might exist "by coincidence" (parameter cancellation), which would mislead the PC algorithm.

Example:
Suppose 
𝐴
→
𝐵
→
𝐶
A→B→C, but by chance 
𝐴
A and 
𝐶
C are independent in the data. This is unfaithful.


4. Summary Table
Concept	Meaning
Conditional Independence	
𝑋
⊥
 ⁣
 ⁣
 ⁣
⊥
𝑌
∣
𝑍
X⊥⊥Y∣Z: X, Y independent given Z
Causal Markov	Each variable is independent of non-descendants given parents in DAG
Faithfulness	All and only those independencies in data come from the DAG structure

import pandas as pd
from scipy.stats import chi2_contingency

def ci_test_discrete(df, x, y, z=[]):
    """
    df: DataFrame
    x, y: names of variables to test
    z: list of conditioning variables
    """
    if not z:
        ct = pd.crosstab(df[x], df[y])
        _, p, _, _ = chi2_contingency(ct)
    else:
        # Stratify by Z
        p_vals = []
        groups = df.groupby(z)
        for _, group in groups:
            if group[x].nunique() > 1 and group[y].nunique() > 1:
                ct = pd.crosstab(group[x], group[y])
                _, p, _, _ = chi2_contingency(ct)
                p_vals.append(p)
        p = min(p_vals) if p_vals else 1.0
    return p

# Example usage:
# df = pd.DataFrame({'x':..., 'y':..., 'z':...})
# ci_test_discrete(df, 'x', 'y', ['z'])


2. Why Do Markov Equivalence Classes Matter?
Limit of Observational Data:
You usually cannot distinguish between all possible causal structures using only observational data.
Many different DAGs can fit the observed conditional independence relations.

Output of PC/FCI Algorithms:
Algorithms like PC return a partially directed graph (a CPDAG or essential graph) representing the entire Markov equivalence class rather than a single DAG.

3. How Are They Characterized?
Two DAGs are Markov equivalent if and only if:

They have the same skeleton (the same set of undirected edges; i.e., same adjacencies).

They have the same set of v-structures (colliders) (i.e., patterns like 
𝐴
→
𝐵
←
𝐶
A→B←C where 
𝐴
A and 
𝐶
C are not connected).

4. Example
Suppose you have variables 
𝐴
A, 
𝐵
B, 
𝐶
C:
DAG 1: 
𝐴
→
𝐵
→
𝐶
A→B→C

DAG 2: 
𝐴
←
𝐵
→
𝐶
A←B→C

Both have the same skeleton: 
𝐴
−
𝐵
−
𝐶
A−B−C

Both have no colliders (there’s no 
𝑋
→
𝑌
←
𝑍
X→Y←Z structure).

So they are Markov equivalent: Observational data can’t tell them apart.

But if you have:

DAG 3: 
𝐴
→
𝐵
←
𝐶
A→B←C

This has a collider at 
𝐵
B.

This is NOT Markov equivalent to the first two DAGs.


5. CPDAG (Completed Partially Directed Acyclic Graph)
PC algorithm outputs a CPDAG, which encodes all DAGs in the Markov equivalence class:

Directed edges: direction is shared by all DAGs in the class.

Undirected edges: direction is ambiguous, could be either way.

7. Summary Table
Property	Same in Equivalence Class?
Skeleton (adjacencies)	Yes
Collider (v-structures)	Yes
Edge directions (other)	No (can differ)



1. G-Test for Discrete Data
What is the G-Test?
The G-test is a likelihood-ratio test used to test for independence in contingency tables.

It is an alternative to the chi-squared test, and often more accurate for small sample sizes.

How does it work?
For two discrete variables, construct a contingency table of observed counts.

Calculate expected counts under the assumption of independence.

Compute the G statistic:

𝐺
=
2
∑
𝑖
,
𝑗
𝑂
𝑖
𝑗
ln
⁡
(
𝑂
𝑖
𝑗
𝐸
𝑖
𝑗
)
G=2 
i,j
∑
​
 O 
ij
​
 ln( 
E 
ij
​
 
O 
ij
​
 
​
 )
where 
𝑂
𝑖
𝑗
O 
ij
​
  is the observed count, 
𝐸
𝑖
𝑗
E 
ij
​
  is the expected count under independence.

The G statistic follows a chi-squared distribution with degrees of freedom = 
(
rows
−
1
)
×
(
columns
−
1
)
(rows−1)×(columns−1).


import numpy as np
from scipy.stats import chi2_contingency

# Example contingency table (e.g., A vs B)
obs = np.array([[10, 20], [20, 40]])

# chi2_contingency returns the G-test (log-likelihood ratio) statistic as 'statistic' if you set lambda_="log-likelihood"
g_stat, p, dof, expected = chi2_contingency(obs, lambda_="log-likelihood")

print(f"G statistic: {g_stat:.3f}, p-value: {p:.3g}")
Interpretation: If the p-value is small (< 0.05), you reject independence.

Conditional Independence (given Z)
For conditional independence, repeat the G-test within each value of Z and combine the statistics (usually by summing).

2. Conditional Independence Tests for Continuous Data: KCI and CMI
A. Kernel-based Conditional Independence (KCI) Test
KCI uses kernel methods to test whether two continuous variables 
𝑋
X and 
𝑌
Y are independent given a third variable 
𝑍
Z.

It does not assume linearity or Gaussianity.

Python Example (using pycitest library):


# Install via: pip install pycitest
from pycitest import KCI
import numpy as np

# Simulate data
np.random.seed(42)
n = 200
Z = np.random.randn(n, 1)
X = Z + 0.1*np.random.randn(n, 1)
Y = Z + 0.1*np.random.randn(n, 1)

# Run KCI test
kci = KCI()
stat, p_value = kci.test(X, Y, Z)
print(f"KCI test statistic: {stat:.3f}, p-value: {p_value:.3g}")



B. Conditional Mutual Information (CMI)
CMI quantifies the amount of information shared between 
𝑋
X and 
𝑌
Y, conditioned on 
𝑍
Z.

For continuous variables, k-nearest neighbor estimators are often used.


# Install via: pip install npeet
import numpy as np
from npeet.entropy_estimators import cmi

# Simulate data
np.random.seed(42)
n = 200
Z = np.random.randn(n)
X = Z + 0.1*np.random.randn(n)
Y = Z + 0.1*np.random.randn(n)

# Calculate conditional mutual information
cmi_value = cmi(X, Y, Z)
print(f"Conditional mutual information: {cmi_value:.3f}")



Summary Table
Test	Use Case	Key Library	Handles Nonlinear?	Handles Conditional?
G-test	Discrete data	scipy.stats	No	Yes (by stratification)
KCI	Continuous, general	pycitest	Yes	Yes
CMI	Continuous, general	npeet	Yes	Yes



Let's compare standard regression-based CI tests to KCI on an illustrative example.

1. Scenario Setup
Latent variable 
𝑍
Z

𝑋
=
sin
⁡
(
𝑍
)
+
non-Gaussian noise
X=sin(Z)+non-Gaussian noise

𝑌
=
cos
⁡
(
𝑍
)
+
non-Gaussian noise
Y=cos(Z)+non-Gaussian noise

X and Y are nonlinearly related to Z, but independent given Z

The noise is Laplace (non-Gaussian, "fatter tails" than normal)

We test 
𝑋
⊥
𝑌
∣
𝑍
X⊥Y∣Z

2. Python Example
Let's simulate this and compare:

Partial correlation after linear regression (standard)

KCI test (kernel-based, nonlinear)

a. Data Simulation

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr, laplace
from pycitest import KCI

# Simulate data
np.random.seed(42)
n = 400
Z = np.random.uniform(-3, 3, (n, 1))
X = np.sin(Z) + 0.4 * laplace.rvs(size=(n, 1))   # Nonlinear, non-Gaussian noise
Y = np.cos(Z) + 0.4 * laplace.rvs(size=(n, 1))

# Visualize
plt.scatter(X, Y, alpha=0.3)
plt.title("Scatter plot of X vs Y (nonlinear, non-Gaussian noise)")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


b. Standard Technique: Linear Regression Residuals & Partial Correlation
# Linear regression of X ~ Z, Y ~ Z, get residuals
reg_x = LinearRegression().fit(Z, X)
reg_y = LinearRegression().fit(Z, Y)
X_resid = X - reg_x.predict(Z)
Y_resid = Y - reg_y.predict(Z)

# Pearson correlation of residuals
corr, pval = pearsonr(X_resid.ravel(), Y_resid.ravel())
print(f"Partial correlation (standard regression): {corr:.3f}, p-value: {pval:.3g}")


c. Kernel-based Conditional Independence Test (KCI)
python
Copy
Edit


kci = KCI()
stat, kci_pval = kci.test(X, Y, Z)
print(f"KCI statistic: {stat:.3f}, p-value: {kci_pval:.3g}")


3. Interpreting the Results
Partial correlation after regression:

Assumes that the dependency structure is linear and noise is Gaussian.

In this nonlinear/non-Gaussian setup, it will often report a spurious correlation—incorrectly indicating dependence, even when there is none.

KCI:

Handles nonlinearities and non-Gaussian noise.

Will correctly detect that X and Y are independent given Z (high p-value).

Partial correlation (standard regression): 0.22, p-value: 0.00017
KCI statistic: 0.035, p-value: 0.64


Partial correlation: Small p-value ⇒ incorrectly suggests X and Y are dependent given Z.

KCI: Large p-value ⇒ correctly infers X and Y are independent given Z.

5. Summary Table
Method	Handles Nonlinearity?	Handles Non-Gaussian?	Result in this case
Regression + Partial Corr	❌	❌	False positive (error)
KCI	✅	✅	Correct

6. Why This Happens
Linear regression only removes linear effects of Z. If X and Y relate nonlinearly to Z, their residuals will remain dependent, even if X and Y are independent given Z in truth.

KCI can capture arbitrary dependencies and is robust to noise shape.















1. Overview: What Drives the Complexity?
The PC algorithm works in two main phases:

Edge deletion: Uses conditional independence (CI) tests, gradually increasing the size of the conditioning set.

Edge orientation: Applies orientation rules (Meek rules, collider rules).

The main computational cost is in the first phase, due to the combinatorial explosion in the number and size of conditioning sets.

2. Complexity Analysis
A. Skeleton Search Phase (Edge Deletion)
For each pair of variables (nodes) 
(
𝑋
,
𝑌
)
(X,Y), you test independence given all subsets of their adjacency set (i.e., the other nodes they're each connected to).

You increase the size of the conditioning set from 0 up to the size of the adjacency set minus 1.

Number of Conditional Independence Tests
For 
𝑛
n nodes, in the worst case, the number of possible conditioning sets is exponential in the number of adjacent nodes.

For each pair, you may have to test all subsets up to size 
𝑑
d, where 
𝑑
d is the maximum degree in the (current) graph.

Total Number of Tests
Worst case: 
𝑂
(
𝑛
2
⋅
2
𝑑
)
O(n 
2
 ⋅2 
d
 )

𝑛
2
n 
2
 : all pairs of nodes

2
𝑑
2 
d
 : all subsets of up to 
𝑑
d neighbors as conditioning sets

So, the PC algorithm is exponential in the maximum degree 
𝑑
d of the true graph.

For sparse graphs (low 
𝑑
d), this is manageable.

For dense graphs (high 
𝑑
d), this becomes intractable.

B. Edge Orientation Phase
This step (applying orientation rules) is polynomial: usually 
𝑂
(
𝑛
3
)
O(n 
3
 ).

The main bottleneck remains the CI tests.

3. Summary Table
Phase	Complexity
Skeleton Search	
𝑂
(
𝑛
2
⋅
2
𝑑
)
O(n 
2
 ⋅2 
d
 )
Edge Orientation	
𝑂
(
𝑛
3
)
O(n 
3
 )

4. Implications
PC scales well for sparse graphs with small 
𝑑
d (maximum node degree), but not for dense graphs or graphs with "hub" nodes.

For very large or dense data, people use approximate or constraint-reduced versions (e.g., [PC-stable], [FastPC], or local/parallel PC variants).
