# AGENTS.md

State-space models, scoring, and oracles for testing data assimilation algorithms in JAX.
No assimilation algorithm lives in the package; see [docs/design.md](docs/design.md) for scope and interface.

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

# Optional editable install
pip install -e '.[dev,pde]'
```

## Code conventions

- One object per file in `jax_da/dynamics/`; `jax_da/__init__.py` re-exports the public API.
- Objects are `flax.struct.dataclass`. Numeric parameters are pytree children; sizes, modes, and `dt` loop counts are `struct.field(pytree_node=False)`.
- Validate numeric parameters in `__post_init__` behind `is_concrete`; validate static fields unguarded.
- Every public method on states starts with `check_event_shape` (`jax_da/_validation.py`); never reshape or broadcast a malformed input.
- States are flat `(..., D)`; leading axes are batch axes; structured layouts are described by `geometry`, not by the array shape.
- Explicit PRNG keys; no global random state.
- Google-style docstrings with shapes in Args/Returns; comments explain why, not what.
- Tests compare against independent references (closed forms, quadrature, Monte Carlo, the Kalman oracle), not stored outputs.
