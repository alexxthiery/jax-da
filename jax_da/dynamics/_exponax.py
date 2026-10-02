"""Lazy, cached access to Exponax spectral steppers (optional ``pde`` extra)."""
import functools

import jax


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


def apply_steps(step, x, n_steps: int, field_shape: tuple[int, ...]):
    """Apply a single-field Exponax ``step`` ``n_steps`` times to flat states ``(..., D)``."""
    batch = x.shape[:-1]
    fields = x.reshape((-1, 1) + field_shape)
    out = jax.vmap(lambda u: jax.lax.fori_loop(0, n_steps, lambda _, v: step(v), u))(fields)
    return out.reshape(batch + (x.shape[-1],))
