# Contributing

## Adding a dynamics

1. Create `jax_da/dynamics/<name>.py` with a `flax.struct.dataclass`; `lorenz96.py` is the template.
   - Fields: numeric parameters are pytree children (they can be swept with `vmap` and differentiated); sizes, `substeps`, and anything that sets a loop count or a solver are `struct.field(pytree_node=False)`.
   - Members: `dim` and `geometry` properties, `flow(x)` over one interval starting with `check_event_shape(x, (self.dim,))`, and for chaotic systems `initial_condition(key)` and `spinup(x, n_steps)`.
   - `flow` must accept any leading batch axes.
   - Validate parameters in `__post_init__`; guard numeric ones with `is_concrete`.
2. Export it from `jax_da/dynamics/__init__.py` and `jax_da/__init__.py`.
3. Add it to `DYNAMICS` in `tests/test_interface.py`, and add `tests/test_<name>.py` with at least one check against an independent reference: a conservation law or energy budget, a fixed point, a hand-computed right-hand side, or a published statistic.
4. Write `docs/<name>.md` (equations, field table, references) and add a row to the README table.
5. If the literature has a standard setting, add a preset to `jax_da/problems.py` and `docs/problems.md`.

## Adding a noise law or an observation operator

Follow the existing classes in `noise.py` or `observations.py`, add the object to `tests/test_interface.py`, and test `log_prob` against SciPy or the operator against hand values and finite differences.

## Tests

```bash
python -m pytest            # default suite
python -m pytest -m pde     # Exponax-backed models
```

Tests compare against independent references, never against stored outputs of the code itself.
Monte Carlo checks set their tolerance from the standard error they expect.
