# Lorenz96

Single-scale Lorenz (1996) model on a ring of `dim` sites:

$$
\frac{dx_i}{dt} = (x_{i+1} - x_{i-2})\, x_{i-1} - x_i + F, \qquad x_{i+D} = x_i .
$$

The quadratic advection term conserves $\tfrac12 \lVert x \rVert^2$, the linear term dissipates, and $F$ forces.
With $F = 8$ and $D = 40$ the leading Lyapunov exponent is about 1.7 per time unit, the climatological mean about 2.3 and standard deviation about 3.6.
One time unit corresponds to roughly 5 days of atmospheric error growth, so `dt = 0.05` is about 6 hours.

```python
import jax_da

model = jax_da.Lorenz96(dim=40, dt=0.05, forcing=8.0)
x1 = model(x0)                          # a map: (..., 40) -> (..., 40), RK4 with `substeps` steps
model.geometry.distances(indices)       # Ring(40): wrapped distances for localization
```

| Field | Default | Kind | Meaning |
|-------|---------|------|---------|
| `dim` | 40 | static | number of sites, at least 4 |
| `dt` | 0.05 | numeric | interval between observations |
| `forcing` | 8.0 | numeric | $F$ |
| `substeps` | 4 | static | RK4 steps per interval |

Preset: `jax_da.problems.lorenz96()` observes every site with $R = I$ (Sakov and Oke 2008); `obs_every=2` or `4` thins the network.

References: Lorenz (1996), Predictability: a problem partly solved; Lorenz and Emanuel (1998).
