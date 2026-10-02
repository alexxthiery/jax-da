# Laws and conditional laws

## Laws

A law is a probability distribution on $\mathbb{R}^d$: the initial law of a model, or the noise inside a conditional law.

| Member | Shapes | Meaning |
|--------|--------|---------|
| `dim` | | dimension $d$ |
| `loc` | `()` or `(d,)` | location; the mean for all laws except `Cauchy` (median) |
| `sample(key, shape=())` | `shape + (d,)` | draws |
| `log_prob(x)` | `(..., d) -> (...)` | log density; raises for `PointMass` |
| `variance()`, `cov()` | `(d,)`, `(d, d)` | raise `NotImplementedError` when the second moment does not exist |

| Law | Construction | Variance |
|-----|--------------|----------|
| `Gaussian` | `Gaussian.isotropic(d, std, loc=0)`, `Gaussian.diagonal(std_vector, loc=0)`, `Gaussian.full(cov, loc=0)` | as given |
| `StudentT` | `StudentT(d, df, scale, loc=0)` or `StudentT.with_std(d, df, std)` | $\text{scale}^2\, \nu / (\nu - 2)$ for $\nu > 2$ |
| `Cauchy` | `Cauchy(d, scale, loc=0)` | undefined |
| `Laplace` | `Laplace(d, scale, loc=0)` or `Laplace.with_std(d, std)` | $2\, \text{scale}^2$ |
| `GaussianMixture` | `GaussianMixture(d, std, outlier_prob, outlier_scale, loc=0)` | $\text{std}^2 (1 - p + p\, k^2)$ |
| `PointMass` | `PointMass(value)` | 0; no density |

All laws except `Gaussian.full` are independent across components; `scale`/`std` may be a scalar or a `(d,)` vector.
Heavy-tailed laws use the textbook `scale`; `with_std` builds one whose standard deviation matches a Gaussian, for comparisons at equal spread.
`GaussianMixture` is Tukey's contaminated Gaussian: each component is $N(\text{loc}, \text{std}^2)$ with probability $1 - p$ and $N(\text{loc}, (k\, \text{std})^2)$ with probability $p$.

## Conditional laws

A conditional law is the law of an output given a state: a transition $x_t \mid x_{t-1}$ or an observation $y_t \mid x_t$.

| Member | Shapes | Meaning |
|--------|--------|---------|
| `in_dim`, `dim` | | dimensions of $x$ and of the output |
| `sample(key, x)` | `(..., in_dim) -> (..., dim)` | a draw of the output |
| `log_prob(out, x)` | `(..., dim), (..., in_dim) -> (...)` | $\log p(\text{out} \mid x)$ |
| `mean(x)` | `(..., in_dim) -> (..., dim)` | $E[\text{out} \mid x]$ |

| Conditional law | Output | Density |
|-----------------|--------|---------|
| `Additive(map, noise=None)` | $\text{map}(x) + \varepsilon$ | `noise.log_prob(out - map(x))`; no density when `noise` is None |
| `Multiplicative(log_scale, noise)` | $e^{g(x)} \odot \varepsilon$, $g$ = `log_scale` | `noise.log_prob(out / s) - sum(log s)`, $s = e^{g(x)}$ |
| `Poisson(log_rate)` | independent counts with rate $e^{r(x)}$ | Poisson log-pmf; counts returned as floats |

`Additive` is the classical data assimilation form and exposes `map` and `noise`, which Kalman-type methods and the oracle read.
`Multiplicative` with a standard Gaussian is $N(0, \operatorname{diag}(s^2))$, the stochastic volatility observation; its log-density uses $g(x)$ directly, so tiny scales never produce $\log 0$.

## Writing your own

Any object with these members works; no base class is needed.
A conditional law for a sign observation that is wrong with probability 0.1:

```python
class SignObservation:
    in_dim, dim = 2, 1

    def mean(self, x):
        return 0.8 * jnp.sign(x[..., :1])

    def sample(self, key, x):
        flip = jax.random.bernoulli(key, 0.1, x.shape[:-1] + (1,))
        return jnp.where(flip, -1.0, 1.0) * jnp.sign(x[..., :1])

    def log_prob(self, y, x):
        return jnp.where(y[..., 0] == jnp.sign(x[..., 0]), jnp.log(0.9), jnp.log(0.1))
```

To pass such a class through `jit` as part of a model argument, make it a `flax.struct.dataclass` (numeric parameters as fields); the contracts are in `jax_da/protocols.py`.
