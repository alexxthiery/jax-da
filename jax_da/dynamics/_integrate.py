"""Fixed-step integration shared by the ODE models."""
import jax


def rk4(rhs, x, dt, substeps: int):
    """Integrate ``dx/dt = rhs(x)`` over ``dt`` with ``substeps`` classical RK4 steps.

    Args:
        rhs: Right-hand side, ``(..., D) -> (..., D)``.
        x: State, ``(..., D)``.
        dt: Total integration time.
        substeps: Number of equal RK4 steps (static).

    Returns:
        State after time ``dt``, same shape as ``x``.
    """
    h = dt / substeps

    def step(_, s):
        k1 = rhs(s)
        k2 = rhs(s + 0.5 * h * k1)
        k3 = rhs(s + 0.5 * h * k2)
        k4 = rhs(s + h * k3)
        return s + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return jax.lax.fori_loop(0, substeps, step, x)


def spin(flow, x, n_steps: int):
    """Apply ``flow`` ``n_steps`` times (``lax.scan``, no stored trajectory)."""
    return jax.lax.scan(lambda s, _: (flow(s), None), x, None, length=n_steps)[0]
