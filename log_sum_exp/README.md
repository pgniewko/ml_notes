# The Log-Sum-Exp Trick in `BCEWithLogitsLoss`

`torch.nn.BCEWithLogitsLoss` combines the sigmoid activation and binary cross-entropy into a single numerically stable operation. The key ingredient is the **log-sum-exp trick**, which avoids overflow and underflow when computing exponentials.

## Binary Cross Entropy

Given a logit (x) and binary target (y \in {0,1}),

[
L(x,y)
======

-y\log\sigma(x)
-(1-y)\log(1-\sigma(x)),
]

where

[
\sigma(x)=\frac{1}{1+e^{-x}}.
]

Substituting the sigmoid gives

[
L(x,y)
======

(1-y)x
+
\log(1+e^{-x}).
]

The only numerically challenging term is

[
\log(1+e^{-x}),
]

which is the **softplus** function.

## Classical Log-Sum-Exp Trick

The classical identity is

[
\log(e^a+e^b)
=============

\max(a,b)
+
\log\left(
e^{a-\max(a,b)}
+
e^{b-\max(a,b)}
\right).
]

Subtracting the maximum ensures that both exponents are non-positive, preventing overflow.

## Applying It to BCE

Observe that

[
\log(1+e^{-x})
==============

\log(e^0+e^{-x}).
]

Applying the log-sum-exp trick with

[
a=0,\qquad b=-x,
]

yields

[
\log(1+e^{-x})
==============

\max(0,-x)
+
\log\left(
e^{-\max(0,-x)}
+
e^{-x-\max(0,-x)}
\right).
]

This is mathematically identical to the original expression but numerically stable.

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

Each term corresponds directly to the mathematical expression:

* `input - input * target` computes
  [
  (1-y)x.
  ]

* `max_val = max(0,-x)` is the shift introduced by the log-sum-exp trick.

* The logarithm computes the stabilized version of
  [
  \log(1+e^{-x}).
  ]

Thus the complete loss is

[
L(x,y)
======

(1-y)x
+
\max(0,-x)
+
\log\left(
e^{-\max(0,-x)}
+
e^{-x-\max(0,-x)}
\right).
]

## Equivalent Compact Form

Since

[
\max(0,-x)=-\min(x,0),
]

the loss is often written as

[
L(x,y)
======

## (1-y)x

\min(x,0)
+
\log\left(1+e^{-|x|}\right).
]

This expression is mathematically equivalent but guarantees that all exponentials are evaluated on non-positive arguments, making it numerically stable even for very large positive or negative logits.

