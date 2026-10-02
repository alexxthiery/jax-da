# AGENTS.md

State-space models, scoring, and oracles for testing data assimilation algorithms in JAX.
No assimilation algorithm lives in the package.
Read in this order: this file, the [README](README.md) (interface table), `jax_da/protocols.py` (component contracts), [docs/design.md](docs/design.md) (scope and decisions).

## Quick reference

- **Package:** `jax_da` (import name), `jax-da` (distribution name)
- **Python:** >=3.10
- **Core deps:** JAX, Flax (`struct.dataclass`), NumPy; optional `pde` extra (Exponax) for KS and Kolmogorov
- **Build:** hatchling
- **License:** MIT

## Commands

```bash
# Tests from a source checkout (pyproject puts the repo root on the path)
python -m pytest

# Exponax-backed models
python -m pytest -m pde

# Mutation check: every plausible bug in tools/mutate.py must be KILLED (about 10 minutes)
python tools/mutate.py

# Editable install (needed to run examples/ directly)
pip install -e '.[dev,pde]'
```

## Architecture

```text
jax_da/
  __init__.py           re-exports the public API
  protocols.py          Dynamics, NoiseLaw, ObservationOperator, Geometry contracts
  ssm.py                StateSpaceModel, Trajectory
  dynamics/             one file per system: linear, lorenz63, lorenz96, lorenz96_two_scale,
                        ks, kolmogorov; _integrate (RK4, spin-up), _exponax (cached steppers)
  noise.py              Gaussian, StudentT, Cauchy, Laplace, GaussianMixture
  observations.py       Selector, Linear, Elementwise
  geometry.py           Unstructured, Ring, Torus2D
  metrics.py            ensemble scores
  oracles.py            KalmanOracle
  problems.py           presets
  _validation.py        shape and parameter checks
docs/                   design.md plus one page per object
examples/               algorithms written against the public API (not part of the package)
tests/                  test_interface.py (every object) plus one file per module
tools/mutate.py         mutation check of the test suite
```

The package never contains an assimilation algorithm; algorithms belong in `examples/` or in consuming projects.

## Code conventions

- One object per file in `jax_da/dynamics/`; `jax_da/__init__.py` re-exports the public API.
- Objects are `flax.struct.dataclass`. Numeric parameters are pytree children; sizes, modes, and `dt` loop counts are `struct.field(pytree_node=False)`.
- Validate numeric parameters in `__post_init__` behind `is_concrete`; validate static fields unguarded.
- Every public method on states starts with `check_event_shape` (`jax_da/_validation.py`); never reshape or broadcast a malformed input.
- States are flat `(..., D)`; leading axes are batch axes; structured layouts are described by `geometry`, not by the array shape.
- Explicit PRNG keys; no global random state.
- Google-style docstrings with shapes in Args/Returns; comments explain why, not what.
- Tests compare against independent references (closed forms, quadrature, Monte Carlo, the Kalman oracle), not stored outputs.

## Testing

Every test protects a claim with an independent oracle; pick from what the existing tests use:

| Object | Oracle used |
|--------|-------------|
| noise laws | SciPy log-densities, 1D quadrature, Monte Carlo moments |
| ODE dynamics | energy budgets, fixed points, hand-computed right-hand sides, RK4 order, Lyapunov exponent |
| PDE dynamics | conserved mean (KS), drag decay of mean vorticity (Kolmogorov), flow over `dt` equals two flows over `dt/2` |
| `StateSpaceModel` | hand Gaussian formulas, exact noiseless alignment, residual laws |
| `KalmanOracle` | brute-force conditioning of the joint Gaussian |
| metrics | pairwise definitions, closed-form Gaussian CRPS, calibrated and under-dispersed synthetic ensembles |
| end to end | `examples/` particle filter vs the oracle (statistical, over 32 runs) |

Monte Carlo tolerances come from the expected standard error, not from trial and error.
After changing behavior, run `tools/mutate.py`; add a mutant for any new behavior worth protecting.

## Gotchas

- `tests/conftest.py` enables float64; library defaults stay float32.
- KS and Kolmogorov parameters are static: changing one builds a new Exponax stepper (cached per configuration).
- `dt` of a PDE model must be a multiple of `dt_inner`.
- `examples/` import `jax_da` as an installed package; tests add the repo root to the path via `pyproject.toml`.
- `Selector.indices` is a static tuple, so it hashes and can drive indexing under `jit`.
