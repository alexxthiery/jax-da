"""Mutation check: apply plausible bugs one at a time and confirm the test suite catches each.

A suite that passes on buggy code is decorative. Each mutant below is a
realistic mistake (off-by-one, wrong sign, dropped term, swapped axis); the
suite runs on a temporary copy of the repository with that one change.

    python tools/mutate.py            # all mutants, about 10 minutes
    python tools/mutate.py M03 M20    # selected mutants

Every mutant must be KILLED. A mutant whose pattern no longer matches the
source fails loudly: update its pattern in the same change that moved the code.
Add a mutant for each new behavior worth protecting.
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (id and description, file under jax_da/, original text, mutated text, needs Exponax)
MUTANTS = [
    ("M01 simulate observes x_t-1", "ssm.py", "self.sample_observation(keys[1], x_next)", "self.sample_observation(keys[1], x)", False),
    ("M02 initial law ignores loc", "laws.py", "return self.loc + self.std * z", "return self.std * z", False),
    ("M03 RK4 weights", "dynamics/_integrate.py", "k1 + 2.0 * k2 + 2.0 * k3 + k4", "k1 + k2 + k3 + k4", False),
    ("M04 L96 index sign", "dynamics/lorenz96.py", "jnp.roll(x, 2, axis=-1)) * jnp.roll(x, 1, axis=-1)", "jnp.roll(x, -2, axis=-1)) * jnp.roll(x, 1, axis=-1)", False),
    ("M05 two-scale coupling sign", "dynamics/lorenz96_two_scale.py", "+ coupling * jnp.repeat(X", "- coupling * jnp.repeat(X", False),
    ("M06 Gaussian logdet dropped", "laws.py", "return (-0.5 * (e / std) ** 2 - jnp.log(std)).sum(-1)", "return (-0.5 * (e / std) ** 2).sum(-1)", False),
    ("M07 StudentT exponent", "laws.py", "- (df + 1) / 2 * jnp.log1p", "- df / 2 * jnp.log1p", False),
    ("M08 mixture outlier inverted", "laws.py", "jax.random.uniform(k_mask, full) < self.outlier_prob", "jax.random.uniform(k_mask, full) > self.outlier_prob", False),
    ("M09 Laplace with_std", "laws.py", "scale=std / math.sqrt(2.0)", "scale=std / 2.0", False),
    ("M10 polynomial exponent", "maps.py", "** (self.degree - 1))", "** self.degree)", False),
    ("M11 Selector.every offset", "maps.py", "tuple(range(offset, in_dim, stride))", "tuple(range(offset + 1, in_dim, stride))", False),
    ("M12 torus axes swapped", "geometry.py", "np.divmod(targets, self.width)", "np.divmod(targets, self.height)", False),
    ("M13 CRPS sort weights", "metrics.py", "(2 * jnp.arange(1, n + 1) - n - 1)", "(2 * jnp.arange(1, n + 1) - n)", False),
    ("M14 spread-skill factor dropped", "metrics.py", "jnp.sqrt((n + 1) / n * var / mse)", "jnp.sqrt(var / mse)", False),
    ("M15 coverage quantile", "metrics.py", "jnp.quantile(ensemble, (1 - level) / 2, axis=-2)", "jnp.quantile(ensemble, 1 - level, axis=-2)", False),
    ("M16 rank uses <=", "metrics.py", "jnp.sum(ensemble < truth[..., None, :], axis=-2)", "jnp.sum(ensemble <= truth[..., None, :], axis=-2) - 1", False),
    ("M17 oracle evidence logdet", "oracles.py", "+ log_det + y.shape[0]", "+ y.shape[0]", False),
    ("M18 oracle smoother gain", "oracles.py", "G = jnp.linalg.solve(P_pred_next, A @ P).T", "G = jnp.linalg.solve(P_pred_next, A @ P)", False),
    ("M19 oracle ignores Q", "oracles.py", "P_pred = A @ P @ A.T + Q", "P_pred = A @ P @ A.T", False),
    ("M20 PDE inner step count", "dynamics/_exponax.py", "    return round(n)\n", "    return round(n) - 1\n", True),
    ("M21 Kolmogorov drag sign", "dynamics/kolmogorov.py", "drag=self.drag,", "drag=-self.drag,", True),
    ("M22 preset ignores obs_every", "problems.py", "Selector.every(dim, obs_every), obs_std", "Selector.every(dim, 1), obs_std", False),
    ("M23 transition noise dropped", "conditional.py", "return center + self.noise.sample(key, center.shape[:-1])", "return center", False),
    ("M24 energy score half", "metrics.py", "return term1 - 0.5 * term2", "return term1 - term2", False),
    ("M26 climatology missing 1/(1-|a|^2)", "problems.py", "sigma = q / (1 - np.abs(a) ** 2)", "sigma = q", False),
    ("M27 advection direction flipped", "problems.py", "np.exp(-1j * c * ki * dt))", "np.exp(1j * c * ki * dt))", False),
    ("M28 Nyquist translation factor 1", "problems.py", "np.where(nyquist, np.cos(c * ki * dt),", "np.where(nyquist, 1.0,", False),
    ("M29 Function not vectorized", "maps.py", "out = jnp.vectorize(self.fn, signature=\"(d)->(e)\")(x)", "out = self.fn(x)", False),
    ("M25 oracle drops Q correlations", "oracles.py", "P_pred = A @ P @ A.T + Q", "P_pred = A @ P @ A.T + jnp.diag(jnp.diag(Q))", False),
    ("M30 Additive mean ignores noise loc", "conditional.py", "return self.map(x) if self.noise is None else self.map(x) + self.noise.loc", "return self.map(x)", False),
    ("M31 Multiplicative Jacobian dropped", "conditional.py", "self.noise.log_prob(out * jnp.exp(-log_s)) - log_s.sum(-1)", "self.noise.log_prob(out * jnp.exp(-log_s))", False),
    ("M32 Poisson factorial dropped", "conditional.py", "- jnp.exp(log_rate) - gammaln(out + 1.0)", "- jnp.exp(log_rate)", False),
    ("M33 oracle ignores offset", "oracles.py", "m_pred = A @ m + b", "m_pred = A @ m", False),
    ("M34 Linear dim from leading axis", "maps.py", "return int(self.matrix.shape[-2])", "return int(self.matrix.shape[0])", False),
    ("M35 SV prior not stationary", "problems.py", "sigma / np.sqrt(1 - phi ** 2)", "sigma", False),
]



def run(mutant, work: Path) -> tuple[bool, str]:
    name, rel, old, new, pde = mutant
    shutil.copytree(ROOT, work, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache"))
    path = work / "jax_da" / rel
    text = path.read_text()
    if text.count(old) != 1:
        raise SystemExit(f"{name}: pattern not found exactly once in jax_da/{rel}; update the mutant")
    path.write_text(text.replace(old, new))
    cmd = [sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", *(["-m", "pde"] if pde else [])]
    result = subprocess.run(cmd, cwd=work, capture_output=True, text=True)
    first_failure = next((line for line in result.stdout.splitlines() if line.startswith("FAILED")), "")
    return result.returncode != 0, first_failure


def main(selected: list[str]) -> int:
    mutants = [m for m in MUTANTS if not selected or m[0].split()[0] in selected]
    survivors = []
    for mutant in mutants:
        with tempfile.TemporaryDirectory() as tmp:
            killed, failure = run(mutant, Path(tmp) / "repo")
        print(f"{'KILLED  ' if killed else 'SURVIVED'} {mutant[0]}  {failure[:100]}", flush=True)
        if not killed:
            survivors.append(mutant[0])
    print(f"\n{len(mutants) - len(survivors)}/{len(mutants)} killed")
    return 1 if survivors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
