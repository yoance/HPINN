"""Predefined PDE problems for the H-PINN solvers.

Each problem bundles what the notebooks defined inline: the domain, the
initial/boundary conditions, the PDE residual, the analytical solution used
for validation and for generating inverse-problem measurements, and the true
PDE parameters (lambda) for the identification setting.

Problems included (matching the PoC notebooks in ``notebooks/``):

* ``heat``: u_t = c^2 u_xx on t in [0, 10], x in [0, 6] with
  u(0, x) = x (l - x) and homogeneous Dirichlet boundaries.
  Residual form: r = u_t + lambda * u_xx with true lambda = -c^2 = -1.

* ``kdv``: Korteweg-de Vries single-soliton problem on t in [-1, 1],
  x in [-10, 10] with u(t, x) = -2 sech^2(x - 4t).
  Residual form: r = u_t + lambda_1 u u_x + lambda_2 u_xxx with true
  lambda = [-6, 1].
"""

from dataclasses import dataclass, field
from typing import Callable, List

import numpy as np
import tensorflow as tf

from .pinn_pde_solver import DTYPE


@dataclass
class PDEProblem:
    key: str
    name: str
    description: str
    lb: List[float]                  # lower bounds [tmin, xmin]
    ub: List[float]                  # upper bounds [tmax, xmax]
    lambda_star: List[float]         # true PDE parameters
    lambda_names: List[str]          # display names, e.g. ["lambda_1"]
    derivative_order: int            # highest spatial derivative in residual
    fun_u_0: Callable                # initial condition u(t=tmin, x)
    fun_u_b: Callable                # boundary condition u(t, x on boundary)
    u_exact: Callable                # analytical solution u(t, x)
    residual: Callable = field(repr=False)  # residual(u, u_t, u_x, u_xx, u_xxx, lambd)

    @property
    def lb_tensor(self):
        return tf.constant(self.lb, dtype=DTYPE)

    @property
    def ub_tensor(self):
        return tf.constant(self.ub, dtype=DTYPE)

    def make_get_r(self):
        """Build a ``get_r(t, x, model, fun_r)`` function compatible with
        ``PINN_PDESolver``, computing derivatives up to ``derivative_order``
        with nested gradient tapes (as done in the notebooks)."""
        order = self.derivative_order

        def get_r(t, x, model, fun_r):
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch(x)
                with tf.GradientTape(persistent=True) as tape2:
                    tape2.watch(x)
                    with tf.GradientTape(persistent=True) as tape3:
                        tape3.watch(t)
                        tape3.watch(x)
                        u = model(tf.stack([t[:, 0], x[:, 0]], axis=1))
                    u_t = tape3.gradient(u, t)
                    u_x = tape3.gradient(u, x)
                u_xx = tape2.gradient(u_x, x) if order >= 2 else None
            u_xxx = tape1.gradient(u_xx, x) if order >= 3 else None
            del tape1, tape2, tape3
            return fun_r(u=u, u_t=u_t, u_x=u_x, u_xx=u_xx, u_xxx=u_xxx)

        return get_r

    def make_fun_r(self, lambd=None):
        """Residual closure with fixed parameters (forward problem).

        ``lambd`` defaults to the true parameters ``lambda_star``.
        """
        lam = self.lambda_star if lambd is None else lambd

        def fun_r(u, u_t, u_x, u_xx, u_xxx):
            return self.residual(u, u_t, u_x, u_xx, u_xxx, lam)

        return fun_r

    def make_fun_r_trainable(self, model):
        """Residual closure reading parameters off ``model.lambd``
        (inverse/identification problem)."""

        def fun_r(u, u_t, u_x, u_xx, u_xxx):
            return self.residual(u, u_t, u_x, u_xx, u_xxx, model.lambd)

        return fun_r


# ---------------------------------------------------------------------------
# Heat equation
# ---------------------------------------------------------------------------

_HEAT_L = 6.0
_HEAT_C = 1.0


def _heat_u0(x):
    return x * (_HEAT_L - x)


def _heat_ub(t, x):
    return tf.zeros((tf.shape(t)[0], 1), dtype=DTYPE)


def _heat_exact(t, x):
    """Fourier series solution truncated at 10 terms (as in the notebook)."""
    total = 0
    for n in range(1, 11):
        total += (4 * _HEAT_L ** 2 / (n ** 3 * np.pi ** 3)) \
            * np.sin(n * np.pi * x / _HEAT_L) \
            * np.exp(-n ** 2 * np.pi ** 2 * _HEAT_C ** 2 * t / _HEAT_L ** 2) \
            * (1 - (-1) ** n)
    return total


def _heat_residual(u, u_t, u_x, u_xx, u_xxx, lambd):
    lam = lambd[0] if not np.isscalar(lambd) and hasattr(lambd, "__len__") else lambd
    return u_t + lam * u_xx


# ---------------------------------------------------------------------------
# KdV equation
# ---------------------------------------------------------------------------

def _kdv_u0(x):
    return -2.0 / tf.math.cosh(x) ** 2


def _kdv_exact(t, x):
    if tf.is_tensor(t) or tf.is_tensor(x):
        return -2.0 / tf.math.cosh(x - 4.0 * t) ** 2
    return -2.0 / np.cosh(x - 4.0 * t) ** 2


def _kdv_ub(t, x):
    # Soliton is essentially zero at the domain edges; using the exact
    # solution keeps the boundary condition consistent.
    return _kdv_exact(t, x)


def _kdv_residual(u, u_t, u_x, u_xx, u_xxx, lambd):
    return u_t + lambd[0] * u * u_x + lambd[1] * u_xxx


PROBLEMS = {
    "heat": PDEProblem(
        key="heat",
        name="Heat equation",
        description="u_t = c² u_xx with u(0,x) = x(l-x) and zero Dirichlet "
                    "boundaries. Residual: u_t + λ u_xx, true λ = -1.",
        lb=[0.0, 0.0],
        ub=[10.0, _HEAT_L],
        lambda_star=[-(_HEAT_C ** 2)],
        lambda_names=["λ"],
        derivative_order=2,
        fun_u_0=_heat_u0,
        fun_u_b=_heat_ub,
        u_exact=_heat_exact,
        residual=_heat_residual,
    ),
    "kdv": PDEProblem(
        key="kdv",
        name="KdV equation",
        description="Korteweg–de Vries single soliton u = -2 sech²(x-4t). "
                    "Residual: u_t + λ₁ u u_x + λ₂ u_xxx, "
                    "true λ = [-6, 1].",
        lb=[-1.0, -10.0],
        ub=[1.0, 10.0],
        lambda_star=[-6.0, 1.0],
        lambda_names=["λ₁", "λ₂"],
        derivative_order=3,
        fun_u_0=_kdv_u0,
        fun_u_b=_kdv_ub,
        u_exact=_kdv_exact,
        residual=_kdv_residual,
    ),
}


def get_problem(key: str) -> PDEProblem:
    try:
        return PROBLEMS[key]
    except KeyError:
        raise KeyError(
            f"Unknown problem '{key}'. Available: {sorted(PROBLEMS)}") from None
