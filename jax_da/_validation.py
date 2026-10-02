"""Internal validation of shape metadata and concrete parameter values."""

import jax


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


def is_concrete(value) -> bool:
    """Whether a parameter holds a value Python can compare.

    Numeric parameters are pytree children: crossing a ``jit`` or ``vmap``
    boundary rebuilds the object with tracers (or plain ``object()``
    placeholders) in their place and reruns ``__post_init__``. Validation of
    numeric parameters is guarded by this function and skipped in that case.
    Static fields (sizes, modes) are always concrete and need no guard.

    Args:
        value: A parameter value.

    JAX also rebuilds pytrees with integer axis specs (a ``vmap`` ``in_axes``
    tree built with ``tree.map``), so a plain Python ``int`` counts as a
    placeholder; constructors cast numeric parameters to float.

    Returns:
        False for JAX tracers, plain ``object()`` and ``int`` placeholders, else True.
    """
    return type(value) not in (object, int) and not isinstance(value, jax.core.Tracer)


def placeholder_dims(*objects) -> bool:
    """True when the ``dim``/``in_dim`` of these objects cannot be read.

    That happens only when JAX rebuilds a pytree with placeholder leaves
    (``None`` or integer ``in_axes`` specs, ``object()``) in place of arrays;
    the object was validated when first built from real values, so
    ``__post_init__`` dimension checks skip the rebuild.
    """
    try:
        for obj in objects:
            obj.dim, getattr(obj, "in_dim", None)
    except (AttributeError, TypeError):
        return True
    return False

