"""Lazy, cached access to Exponax spectral steppers (optional ``pde`` extra)."""
import functools

import jax
import jax.numpy as jnp


def require_exponax():
    try:
        import exponax
    except ImportError as error:
        raise ImportError(
            "This model needs Exponax: pip install 'jax-da[pde]' (or pip install exponax)") from error
    return exponax


@functools.lru_cache(maxsize=None)
def stepper(name: str, **kwargs):
    """Build ``exponax.stepper.<name>(**kwargs)`` once per configuration.

    Building computes exponential-integrator coefficients, which is too slow
    to repeat on every call outside ``jit``. The first build may happen while
    tracing; ``ensure_compile_time_eval`` keeps the cached coefficients
    concrete so no tracer leaks into the cache.
    """
    with jax.ensure_compile_time_eval():
        return getattr(require_exponax().stepper, name)(**kwargs)


def inner_steps(dt: float, dt_inner: float) -> int:
    """Number of solver steps covering ``dt``; raises unless ``dt`` is a positive multiple of ``dt_inner``."""
    n = dt / dt_inner
    if abs(n - round(n)) > 1e-9 or round(n) < 1:
        raise ValueError(f"dt={dt} must be a positive multiple of dt_inner={dt_inner}")
    return round(n)


def apply_steps(step, x, n_steps: int, field_shape: tuple[int, ...]):
    """Apply a single-field Exponax ``step`` ``n_steps`` times to flat states ``(..., D)``."""
    def one_state(state):
        field = state.reshape((1,) + field_shape)
        return jax.lax.fori_loop(0, n_steps, lambda _, u: step(u), field).reshape(-1)

    return jnp.vectorize(one_state, signature="(d)->(d)")(x)
