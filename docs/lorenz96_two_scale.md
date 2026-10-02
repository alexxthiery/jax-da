# Lorenz96TwoScale

Two-scale Lorenz (1996) model: $K$ slow variables $X_k$ on a ring, each coupled to $J$ fast variables $Y_{j,k}$:

$$
\begin{aligned}
\frac{dX_k}{dt} &= (X_{k+1} - X_{k-2}) X_{k-1} - X_k + F - \frac{hc}{b} \sum_{j} Y_{j,k}, \\
\frac{dY_{j,k}}{dt} &= c b\, Y_{j+1,k} (Y_{j-1,k} - Y_{j+2,k}) - c\, Y_{j,k} + \frac{hc}{b} X_k .
\end{aligned}
$$

The fast variables form one ring of length $KJ$, so $Y_{J,k}$ neighbors $Y_{1,k+1}$.
The state is `[X_1..X_K, Y_{1,1}..Y_{J,K}]`, shape `(K + K J,)`, and the geometry is `Unstructured`.
Coupling and advection conserve energy: $\frac{d}{dt}\tfrac12(\lVert X \rVert^2 + \lVert Y \rVert^2) = -\lVert X \rVert^2 + F \sum_k X_k - c \lVert Y \rVert^2$.

Typical use is a structural model-error study: simulate the truth with this model, observe the slow block, and assimilate with `Lorenz96(dim=K)`.

```python
truth = jax_da.problems.lorenz96_two_scale()          # K = 8, J = 32, F = 20, h = 1, c = b = 10
forecast_model = jax_da.Lorenz96(dim=8, forcing=20.0)
```

| Field | Default | Kind | Meaning |
|-------|---------|------|---------|
| `n_slow`, `n_fast` | 8, 32 | static | $K$, $J$ |
| `dt` | 0.05 | numeric | interval between observations |
| `forcing`, `coupling_h`, `time_scale_c`, `amplitude_b` | 20, 1, 10, 10 | numeric | $F, h, c, b$ |
| `substeps` | 50 | static | RK4 steps per interval; the fast scale is $c$ times quicker |

References: Lorenz (1996); Wilks (2005), Effects of stochastic parametrizations in the Lorenz '96 system.
