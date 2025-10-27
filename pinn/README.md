# Physics-Informed Neural Networks (PINNs 101)

## 🔍 Overview

Physics-Informed Neural Networks (PINNs) are a class of neural networks that integrate the governing laws of physics—typically expressed as differential equations—directly into the learning process.  
Unlike purely data-driven models, PINNs combine **data** and **physical priors** to learn solutions that are both accurate and physically consistent.

PINNs are particularly effective for solving partial differential equations (PDEs), modeling dynamical systems, and performing **inverse problems** where unknown parameters of a physical process are inferred from limited or noisy data.

---

## ⚙️ How Physics Is Introduced into Neural Networks

When training a neural model, physics can be incorporated in **three main ways**: through **data bias**, **inductive bias**, and **learning bias**.  
Each plays a distinct role in constraining or guiding the model toward physically meaningful behavior.

### 1. Data Bias

Physics can be introduced implicitly through the **data** used for training.  
If the training dataset is generated or augmented using known physical laws—e.g., simulation outputs or synthetic data that obeys conservation laws—the network will naturally learn to follow those constraints.

**Example:**  
Providing temperature profiles generated from the heat equation helps the model learn temperature diffusion patterns even without explicitly enforcing the PDE during training.

**Pros:**  
- Straightforward and flexible  
- Works well when large amounts of high-quality simulation data are available  

**Cons:**  
- Model learns physics only indirectly  
- May not generalize outside the data distribution  

---

### 2. Inductive Bias

Inductive bias encodes physics directly into the **architecture or representation** of the model.  
This can mean designing the network structure, activation functions, or features to reflect known symmetries, conservation principles, or invariants of the physical system.

**Example:**  
Using coordinate embeddings that enforce boundary periodicity, or network layers that conserve energy or mass by design.

**Pros:**  
- Physics becomes an inherent property of the model  
- Reduces the need for explicit regularization  

**Cons:**  
- Requires problem-specific architectural design  
- Harder to adapt to new physical systems  

---

### 3. Learning Bias

Learning bias introduces physics as an explicit **constraint during optimization**—that is, through the **loss function**.  
This is the hallmark of most modern PINNs.

The loss combines standard data-fitting terms with **physics residuals** derived from governing equations (e.g., Navier–Stokes, Poisson, or heat equation).  
Automatic differentiation computes derivatives of the neural network output, allowing the PDE residuals to be evaluated directly.

\[
\mathcal{L} = \mathcal{L}_{data} + \lambda \mathcal{L}_{physics}
\]

Here, the physics-based term enforces that the network’s predictions satisfy the underlying laws, while the data term ensures agreement with measurements.

**Pros:**  
- Enforces physical consistency even with few data points  
- Highly flexible and problem-agnostic  
- Enables solving forward and inverse problems  

**Cons:**  
- Can be computationally expensive  
- Sensitive to the relative weighting of loss terms  

---

## 🧠 Problem Description: 1D Heat Equation

This repository demonstrates a PINN applied to the **1D heat conduction problem**, a classic PDE that models how temperature diffuses over time.

**Governing equation:**
\[
u_t = \alpha\,u_{xx}, \quad x \in [0,1], \, t \in [0,1]
\]

**Initial condition:**
\[
u(x,0) = \sin(\pi x)
\]

**Boundary conditions:**
\[
u(0,t) = u(1,t) = 0
\]

**Analytic solution:**
\[
u(x,t) = e^{-\alpha\pi^2 t}\sin(\pi x)
\]

The neural network learns an approximation \(u_\theta(x,t)\) by minimizing a combined loss that includes:
- **Initial condition loss**  
- **Boundary condition loss**  
- **Physics (PDE residual) loss**  

---

## 🧩 Key Takeaways

- PINNs merge deep learning with scientific computing by enforcing physics through data, architecture, or optimization.  
- Among these, **data bias** and **learning bias** are the most commonly used because they provide flexibility across diverse physical systems.  
- This repository provides a minimal yet complete implementation that can be extended to higher-dimensional PDEs, stochastic systems, or parameter estimation tasks.

---

## 📘 Further Reading

- Raissi, M., Perdikaris, P., & Karniadakis, G.E. (2019). *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.*  
  **Journal of Computational Physics**, 378, 686–707.  
- Karniadakis G.E. et al. (2021). *Physics-Informed Machine Learning.* **Nature Reviews Physics**, 3, 422–440.

---

© 2025 Your Name. Licensed under the MIT License.
