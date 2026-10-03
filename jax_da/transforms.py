"""Model transforms: build a new state-space model from an existing one."""

from jax_da.conditional import Lagged, Precomposed
from jax_da.laws import History
from jax_da.maps import Selector
from jax_da.ssm import StateSpaceModel


def delayed(ssm: StateSpaceModel, lags: int) -> StateSpaceModel:
    """Delayed-observation version of ``ssm``: ``y_t`` observes ``x_{t-lags}``.

    The state is the stacked history ``u_t = (x_t, x_{t-1}, ..., x_{t-lags})``, newest
    block first. The transition draws the newest block from ``ssm.transition`` and
    shifts the rest (``Lagged``); the observation applies ``ssm.observation`` to the
    oldest block (``Precomposed``); ``u_0`` holds the first ``lags + 1`` states of a
    trajectory of ``ssm`` (``History``). Hence ``y_t`` has the law of the base model's
    ``y_t``: the oldest block of ``u_t`` is the base state ``x_t``, and the filtering
    law of that block given ``y_1..y_t`` is the base filter. Fresh noise reaches the
    observations only after ``lags`` transitions, which makes particle methods select
    on inherited randomness. Works for any model; linear-Gaussian models stay solvable
    by ``KalmanOracle``.

    Args:
        ssm: Base model on ``R^D``.
        lags: Delay ``L >= 1``.

    Returns:
        A model on ``R^{(L + 1) D}`` with the same observation space. Its geometry is
        ``Unstructured``: the base geometry describes one block, not the stack.
    """
    transition = Lagged(ssm.transition, lags)  # validates lags before it is used below
    D = ssm.state_dim
    oldest = Selector((lags + 1) * D, tuple(range(lags * D, (lags + 1) * D)))
    return StateSpaceModel(
        initial=History(ssm.initial, ssm.transition, lags),
        transition=transition,
        observation=Precomposed(ssm.observation, oldest),
    )
