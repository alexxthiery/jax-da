# jax-da

State-space models, scoring, and oracles for testing data assimilation algorithms in JAX.

jax-da simulates benchmark data assimilation problems (Lorenz-63, Lorenz-96, two-scale Lorenz-96, Kuramoto-Sivashinsky, Kolmogorov flow, linear-Gaussian), exposes their transition and observation densities, scores ensembles against the truth, and computes exact Kalman answers for linear-Gaussian models.
It contains no assimilation algorithm: bring your own filter.

Status: under construction. The interface is specified in [docs/design.md](docs/design.md).

## Installation

```bash
git clone <repository-url> jax-da
cd jax-da
pip install -e '.[dev]'        # add ,pde for Kuramoto-Sivashinsky and Kolmogorov flow
```
