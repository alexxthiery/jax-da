# Contributing

## Adding a dynamics

For a one-off model, wrap a JAX function with `Function` (see `docs/function_models.md`); write a class when the model should expose structure or have differentiable parameters.

1. Create `jax_da/dynamics/<name>.py` with a `flax.struct.dataclass` satisfying `jax_da.protocols.Map` (with `in_dim == dim`); `tanh_squared.py` is the short template, `lorenz96.py` the ODE one.
   - Fields: numeric parameters are pytree children (they can be swept with `vmap` and differentiated); sizes, `substeps`, and anything that sets a loop count or a solver are `struct.field(pytree_node=False)`.
   - Members: `dim`, `in_dim`, and `geometry` properties; `__call__(x)` over one interval starting with `check_event_shape(x, (self.dim,))`; for chaotic systems `initial_condition(key)` and `spinup(x, n_steps)`.
   - `__call__` must accept any leading batch axes; read dimensions from trailing array axes.
   - Validate parameters in `__post_init__`; guard numeric ones with `is_concrete`.
2. Export it from `jax_da/dynamics/__init__.py` and `jax_da/__init__.py`.
3. Add it to `MAPS` in `tests/test_interface.py`, and add tests with at least one check against an independent reference: a conservation law or energy budget, a fixed point, a hand-computed right-hand side, or a published statistic.
4. Write `docs/<name>.md` (equations, field table, references) and add a row to the README table.
5. If the literature has a standard setting, add a preset to `jax_da/problems.py` and `docs/problems.md`.

## Adding a map, law, or conditional law

Follow the existing classes in `maps.py`, `laws.py`, or `conditional.py`, add the object to `MAPS`, `LAWS`, or `CONDITIONAL` in `tests/test_interface.py`, and test `log_prob` against SciPy or a map against hand values and finite differences.

## Tests

```bash
python -m pytest            # default suite
python -m pytest -m pde     # Exponax-backed models
```

Tests compare against independent references, never against stored outputs of the code itself.
Monte Carlo checks set their tolerance from the standard error they expect.
After a behavior change, run `python tools/mutate.py`; every mutant must be killed, and new behavior gets a new mutant.
