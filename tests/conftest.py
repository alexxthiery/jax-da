"""Shared test setup."""
import jax

# Reference comparisons (SciPy, NumPy, the Kalman oracle) need float64.
jax.config.update("jax_enable_x64", True)
