# KalmanOracle

Exact filtering, smoothing, and evidence for linear-Gaussian state-space models: `Additive` transition and observation with a `Linear` or `Selector` map and `Gaussian` (or no) noise, and a `Gaussian` or `PointMass` initial law.
Affine offsets are allowed: the model is $x_t = A x_{t-1} + b + \eta_t$, $y_t = H x_t + c + \varepsilon_t$, where $b$ and $c$ collect the `Linear.offset` and the noise `loc`.
It is the reference for testing a method on the one model class where the answer is known.

```python
import jax
jax.config.update("jax_enable_x64", True)   # compare methods to the oracle in float64

ssm = jax_da.problems.linear_gaussian_full()    # dense A and H, full Q, R, P0
traj = ssm.simulate(key, 50)
oracle = jax_da.KalmanOracle.from_ssm(ssm)
f = oracle.filter(traj.observations)        # FilterResult
s = oracle.smooth(traj.observations)        # SmootherResult
```

| Result | Fields |
|--------|--------|
| `FilterResult` | `means (T, D)` and `covs (T, D, D)` of $x_t \mid y_{1:t}$; `predicted_means`, `predicted_covs` of $x_t \mid y_{1:t-1}$; `log_evidence` $= \log p(y_{1:T})$ |
| `SmootherResult` | `means (T, D)` and `covs (T, D, D)` of $x_t \mid y_{1:T}$ (Rauch-Tung-Striebel) |

`from_ssm` raises `ValueError` for a model that is not linear-Gaussian.

When comparing a Monte Carlo method to the oracle, test statistically: average independent runs and compare with standard errors, and compare the evidence ratio $\exp(\log \hat Z - \log Z)$, which is unbiased for particle filters, rather than $\log \hat Z$.
`tests/test_examples.py` does this for the bootstrap particle filter in `examples/`.
