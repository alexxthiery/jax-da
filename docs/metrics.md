# Metrics

Scores of an ensemble against the truth, in `jax_da.metrics`.
An ensemble has shape `(..., N, D)` and the truth `(..., D)`, where `...` is usually time.
Per-time scores return `(...)`; average them over the window you score.

| Function | Returns | Definition |
|----------|---------|------------|
| `rmse(ensemble, truth)` | `(...)` | $\sqrt{\operatorname{mean}_d (\bar x_d - y_d)^2}$ |
| `spread(ensemble)` | `(...)` | $\sqrt{\operatorname{mean}_d \operatorname{Var}_n x_{nd}}$ with `ddof=1` |
| `crps(ensemble, truth, fair=False)` | `(...)` | $E\lvert X - y \rvert - \tfrac12 E \lvert X - X' \rvert$, averaged over components |
| `crps_gaussian(mean, std, truth)` | `(...)` | closed form for independent Gaussian forecasts |
| `energy_score(ensemble, truth)` | `(...)` | $E\lVert X - y \rVert - \tfrac12 E\lVert X - X' \rVert$; equals CRPS when $D = 1$ |
| `coverage(ensemble, truth, level=0.9)` | `(...)` | fraction of components inside the central empirical `level` interval |
| `spread_skill_ratio(ensemble, truth)` | scalar | $\sqrt{\tfrac{N+1}{N}\, \overline{\operatorname{Var}} / \overline{(\bar x - y)^2}}$ over all leading axes |
| `rank_histogram(ensemble, truth)` | `(N + 1,)` | counts of the number of members below the truth |

`crps(..., fair=True)` uses $N(N-1)$ instead of $N^2$ in the second term: it is unbiased for the CRPS of the distribution the members are drawn from, so ensembles of different sizes compare fairly (Ferro 2014).

The $(N+1)/N$ factor in `spread_skill_ratio` makes a calibrated ensemble (truth exchangeable with the members) score 1 at every $N$ (Fortin et al. 2014).
Below 1 means under-dispersed, above 1 over-dispersed.
A flat `rank_histogram` means calibrated; U-shaped under-dispersed; dome-shaped over-dispersed; sloped biased.

`energy_score` needs $O(N^2 D)$ memory per leading index.
