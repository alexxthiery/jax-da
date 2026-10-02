# AGENTS.md

State-space models, scoring, and oracles for testing data assimilation algorithms in JAX.
No assimilation algorithm lives in the package.
Read in this order: this file, the [README](README.md) (interface table), `jax_da/protocols.py` (the Map, Law, and ConditionalLaw contracts), [docs/design.md](docs/design.md) (scope and decisions).

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

# Mutation check: every plausible bug in tools/mutate.py must be KILLED (one test-suite run per mutant)
python tools/mutate.py

# Editable install (needed to run examples/ directly)
pip install -e '.[dev,pde]'
```

## Architecture

```text
jax_da/
  __init__.py           re-exports the public API
  protocols.py          Map, Law, ConditionalLaw, Geometry contracts
  ssm.py                StateSpaceModel (initial, transition, observation, geometry), Trajectory
  maps.py               Linear (affine), Selector, Elementwise, Function
  laws.py               Gaussian, StudentT, Cauchy, Laplace, GaussianMixture, PointMass
  conditional.py        Additive, Multiplicative, Poisson
  dynamics/             nonlinear maps, one file each: lorenz63, lorenz96, lorenz96_two_scale,
                        ks, kolmogorov, tanh_squared; _integrate (RK4, spin-up), _exponax (cached steppers)
  geometry.py           Unstructured, Ring, Torus2D
  metrics.py            ensemble scores
  oracles.py            KalmanOracle (affine linear-Gaussian)
  problems.py           presets
  _validation.py        shape checks, placeholder-aware parameter checks
docs/                   design.md plus one page per object
examples/               algorithms written against the public API (not part of the package)
tests/                  test_interface.py (every object) plus one file per module
tools/mutate.py         mutation check of the test suite
```

The package never contains an assimilation algorithm; algorithms belong in `examples/` or in consuming projects.

## Code conventions

- One object per file in `jax_da/dynamics/`; `jax_da/__init__.py` re-exports the public API.
- Objects are `flax.struct.dataclass`. Numeric parameters are pytree children; sizes, modes, and `dt` loop counts are `struct.field(pytree_node=False)`.
- Fail early and loud: every mistake raises `ValueError` with a message naming it; never return NaN or a plausible wrong answer. Shapes are checked always; values (`check_positive`, `check_finite`, symmetry, counts) when `is_concrete`, that is outside `jit`. Add each new check to `tests/test_fail_loud.py` with a `match=` on its message.
- Validation in `__post_init__` must survive JAX rebuilds: skip non-numeric leaves (`is_numeric`), guard dimension checks with `placeholder_dims`, read dimensions from trailing axes (`shape[-1]`), and normalize per-component parameters with `per_component`.
- Every public method on states starts with `check_event_shape` (`jax_da/_validation.py`); never reshape or broadcast a malformed input.
- States are flat `(..., D)`; leading axes are batch axes; structured layouts are described by `geometry`, not by the array shape.
- Explicit PRNG keys; no global random state.
- Google-style docstrings with shapes in Args/Returns; comments explain why, not what.
- Tests compare against independent references (closed forms, quadrature, Monte Carlo, the Kalman oracle), not stored outputs.

## Testing

Every test protects a claim with an independent oracle; pick from what the existing tests use:

| Object | Oracle used |
|--------|-------------|
| laws | SciPy log-densities (with nonzero `loc`), 1D quadrature, Monte Carlo moments, finite-difference gradients |
| conditional laws | SciPy densities of the scaled noise and of Poisson counts, sampling moments |
| ODE dynamics | energy budgets, fixed points, hand-computed right-hand sides, RK4 order, Lyapunov exponent |
| PDE dynamics | conserved mean (KS), drag decay of mean vorticity (Kolmogorov), flow over `dt` equals two flows over `dt/2` |
| `StateSpaceModel` | hand Gaussian formulas, exact noiseless alignment, residual laws |
| `KalmanOracle` | brute-force conditioning of the joint Gaussian built from raw constants (Gaussian and point-mass starts); Selector equals its dense matrix |
| presets | documented settings (matrices, networks, noise levels) checked exactly; stationarity and climatology statistics |
| misuse | `tests/test_fail_loud.py`: one realistic mistake per case, `match=` on the message |
| metrics | pairwise definitions, closed-form Gaussian CRPS, calibrated and under-dispersed synthetic ensembles |
| end to end | `examples/` particle filter vs the oracle (statistical, over 32 runs) |

Monte Carlo tolerances come from the expected standard error, not from trial and error.
A test that only checks shape, finiteness, or "runs" is not enough for a behavior; pair it with one of the oracles above.

Proving a test is useful (do this for every new behavior):

1. Add a mutant to `tools/mutate.py`: the most plausible wrong implementation of the behavior.
2. Run `python tools/mutate.py <id>` and see it SURVIVE; this proves the gap.
3. Write the test; run the mutant again and see it KILLED by that test (the first FAILED line names it).
4. A kill by an unrelated test is incidental, not protection; check by rerunning with that test file ignored.
5. Before committing a behavior change, run the full `python tools/mutate.py`: every mutant must be killed.

## Gotchas

- `tests/conftest.py` enables float64; library defaults stay float32.
- KS and Kolmogorov parameters are static: changing one builds a new Exponax stepper (cached per configuration).
- `dt` of a PDE model must be a multiple of `dt_inner`.
- `examples/` import `jax_da` as an installed package; tests add the repo root to the path via `pyproject.toml`.
- `Selector.indices` is a static tuple, so it hashes and can drive indexing under `jit`.
