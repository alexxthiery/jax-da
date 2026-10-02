# Noise laws

Zero-mean additive noise for model error and observation error, in `jax_da.noise`.

| Member | Shapes | Meaning |
|--------|--------|---------|
| `dim` | | dimension $d$ |
| `sample(key, shape=())` | `shape + (d,)` | draws |
| `log_prob(e)` | `(..., d) -> (...)` | log density |
| `variance()` | `(d,)` | per-component variance; raises `NotImplementedError` if infinite |
| `cov()` | `(d, d)` | covariance; raises `NotImplementedError` if undefined |

| Law | Construction | Variance |
|-----|--------------|----------|
| `Gaussian` | `Gaussian.isotropic(d, std)`, `Gaussian.diagonal(std_vector)`, `Gaussian.full(cov)` | as given |
| `StudentT` | `StudentT(d, df, scale)` or `StudentT.with_std(d, df, std)` | $\text{scale}^2\, \nu / (\nu - 2)$ for $\nu > 2$ |
| `Cauchy` | `Cauchy(d, scale)` | undefined |
| `Laplace` | `Laplace(d, scale)` or `Laplace.with_std(d, std)` | $2\, \text{scale}^2$ |
| `GaussianMixture` | `GaussianMixture(d, std, outlier_prob, outlier_scale)` | $\text{std}^2 (1 - p + p\, k^2)$ |

All laws except `Gaussian.full` are independent across components; `scale`/`std` may be a scalar or a `(d,)` vector.
Heavy-tailed laws use the textbook `scale`; `with_std` builds one whose standard deviation matches a Gaussian, for comparisons at equal spread.
`GaussianMixture` is Tukey's contaminated Gaussian: each component is $N(0, \text{std}^2)$ with probability $1 - p$ and $N(0, (k\, \text{std})^2)$ with probability $p$.
