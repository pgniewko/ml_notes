# The Log-Sum-Exp trick in `BCEWithLogitsLoss`

`torch.nn.BCEWithLogitsLoss` combines the sigmoid activation and binary cross-entropy into a single numerically stable operation. The key ingredient is the **log-sum-exp trick**, which avoids overflow and underflow when computing exponentials.

## Binary Cross Entropy

Given a logit value $x$ and binary target $y \in [0,1]$.


$$
L(x,y) = -y\cdot\log\sigma(x)- (1-y)\cdot\log(1-\sigma(x))
$$

where

$$
\sigma(x)=\frac{1}{1+e^{-x}}.
$$

Substituting the sigmoid gives

$$
L(x,y) = (1-y)x + \log\left(1+e^{-x}\right).
$$


The only numerically challenging term is

$$
{\rm softplus}(x) = \log\left(1+e^{-x}\right).
$$

---

## Classical Log-Sum-Exp Trick

The classical identity is

$$
\log\left(e^a+e^b\right) = \max(a,b) + \log\left(e^{a-\max(a,b)} + e^{b-\max(a,b)}\right).
$$

Subtracting the maximum ensures that every exponent is non-positive, preventing overflow.

---

## Applying it to BCE

Observe that

$$
\log\left(1+e^{-x}\right)=\log\left(e^0+e^{-x}\right).
$$

Applying the `log-sum-exp` trick with

$$
a=0,
\qquad
b=-x,
$$

gives

$$
\log\left(1+e^{-x}\right)=\max(0,-x)+\log\left(e^{-\max(0,-x)}+e^{-x-\max(0,-x)}\right).
$$

This expression is mathematically identical to the original one but is numerically stable.

---

## PyTorch Implementation

PyTorch implements this as

```python
max_val = (-input).clamp(min=0)

loss = input - input * target \
     + max_val \
     + torch.log(
         torch.exp(-max_val) +
         torch.exp(-input - max_val)
     )
```

Each term corresponds directly to the mathematics:

- `input - input * target` computes

$$
  (1-y)x.
$$

- `max_val = max(0, -x)` is the shift introduced by the log-sum-exp trick.

- The logarithm computes the stabilized version of

$$
  \log\left(1+e^{-x}\right).
$$

Putting everything together,

$$
L(x,y)=(1-y)x+\max(0,-x)+\log\left(e^{-\max(0,-x)}+e^{-x-\max(0,-x)}\right).
$$

---

## Equivalent Compact Form

Since

$$
\max(0,-x)=-\min(x,0),
$$

the loss is often written as

$$
L(x,y)=(1-y)x-\min(x,0)+\log\left(1+e^{-|x|}\right).
$$

This form is mathematically equivalent and guarantees that every exponential is evaluated on a non-positive argument, making it numerically stable even for very large positive or negative logits.
