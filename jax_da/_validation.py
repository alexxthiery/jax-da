"""Internal validation: fail early and loud, without breaking JAX transformations.

Policy (see docs/design.md):

- shapes are always checked, including under ``jit`` (shapes are static);
- values (positivity, finiteness, symmetry, integer counts) are checked when
  concrete, that is outside ``jit``; inside ``jit`` they cannot be inspected;
- ``__post_init__`` also runs when JAX rebuilds a pytree, with placeholder
  leaves (``object()``, ``None``, or tuples from ``tree.map(jnp.shape, ...)``)
  or with batched leaves of shape ``(..., dim)``; checks skip placeholders and
  read dimensions from trailing axes, so rebuilds and batching keep working.
"""

import jax
import jax.numpy as jnp
import numpy as np

_NUMERIC = (int, float, np.ndarray, np.generic, jax.Array)


def check_event_shape(x: jax.Array, event_shape: tuple[int, ...], name: str = "x") -> None:
    """Require the trailing axes of ``x`` to match one event.

    Leading batch axes are unrestricted, including axes of length zero.
    Only shape metadata is compared, so under ``jit`` the check runs while
    tracing and adds nothing to the compiled computation. It must not be
    skipped for tracers.

    Args:
        x: Array of shape ``batch_shape + event_shape``.
        event_shape: Expected trailing dimensions, e.g. ``(state_dim,)``.
        name: Argument name used in the error message.

    Raises:
        ValueError: If ``x`` has too few axes or its trailing axes differ.
    """
    n = len(event_shape)
    if x.ndim < n or tuple(x.shape[x.ndim - n:]) != tuple(event_shape):
        raise ValueError(
            f"Expected {name}.shape[-{len(event_shape)}:] == {tuple(event_shape)}, "
            f"got shape {tuple(x.shape)}."
        )


def is_numeric(value) -> bool:
    """A number or array (concrete or traced), as opposed to a pytree placeholder."""
    return isinstance(value, _NUMERIC)


def is_concrete(value) -> bool:
    """A number or array whose values Python can inspect: numeric and not a JAX tracer.

    Placeholders that JAX puts in rebuilt pytrees (``object()``, ``None``,
    tuples) are not numeric and return False, so value checks skip them.
    Plain Python ints are concrete: ``StudentT(3, df=-1)`` must raise.
    """
    return is_numeric(value) and not isinstance(value, jax.core.Tracer)


def check_finite(value, name: str) -> None:
    """Raise if a concrete value contains NaN or inf."""
    if is_concrete(value) and not bool(np.all(np.isfinite(np.asarray(value)))):
        raise ValueError(f"{name} must be finite; it contains NaN or inf (shape {np.shape(value)})")


def check_positive(value, name: str) -> None:
    """Raise unless every entry of a concrete value is finite and positive."""
    if is_concrete(value):
        arr = np.asarray(value)
        if not (np.all(np.isfinite(arr)) and np.all(arr > 0)):
            raise ValueError(f"{name} must be positive and finite, got {value}")


def per_component(obj, name: str, dim: int) -> None:
    """Normalize the parameter ``obj.<name>`` to shape ``(..., dim)`` and check its shape.

    A scalar is broadcast to ``(dim,)``, so a law's ``loc`` or ``scale`` always
    has the state's trailing shape; anything else must end in ``dim``. A
    length-1 vector for ``dim > 1`` is rejected rather than silently broadcast.
    Call it from ``__post_init__`` of a ``flax.struct.dataclass``.
    """
    value = getattr(obj, name)
    if not is_numeric(value):
        return
    shape = np.shape(value)
    if shape == ():
        object.__setattr__(obj, name, jnp.broadcast_to(jnp.asarray(value, dtype=float), (dim,)))
    elif shape[-1] != dim:
        raise ValueError(f"{name} must be a scalar or have trailing shape ({dim},), got shape {shape}")


def require_map(value, role: str) -> None:
    """Raise if ``value`` is a plain function (or another callable) instead of a map."""
    from jax_da.protocols import Map

    if callable(value) and not isinstance(value, Map):
        raise ValueError(f"{role} must be a map with in_dim, dim, and __call__; wrap a plain "
                         f"function with jd.Function(fn, in_dim, dim)")


def placeholder_dims(*objects) -> bool:
    """True when the ``dim``/``in_dim`` of these objects cannot be read.

    That happens only when JAX rebuilds a pytree with placeholder leaves in
    place of arrays; the object was validated when first built from real
    values, so ``__post_init__`` dimension checks skip the rebuild.
    """
    try:
        for obj in objects:
            obj.dim, getattr(obj, "in_dim", None)
    except (AttributeError, TypeError, IndexError):
        return True
    return False
