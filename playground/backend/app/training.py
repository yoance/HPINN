"""Training service: builds and trains PINN models from an API request.

This module bridges the ``hpinn`` core library and the web API. It wraps the
library solvers with progress reporting and cooperative cancellation, and
produces JSON-friendly solution grids for the frontend.
"""

import os
import time

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")  # Keras 2 API (see hpinn)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

from hpinn import (
    DTYPE,
    PINN_NeuralNet,
    PINN_PDESolver,
    PINN_IdentificationNet,
    PINN_IdentificationSolver,
    Hadamard_IdentificationNet,
    Hadamard_IdentificationSolver,
    generate_training_data,
    generate_collocation_points,
    generate_training_data_inverse,
    get_problem,
)

tf.keras.backend.set_floatx(DTYPE)

GRID_NT = 80  # solution grid resolution (t axis)
GRID_NX = 80  # solution grid resolution (x axis)


class TrainingCancelled(Exception):
    pass


class _MonitorMixin:
    """Replaces the solvers' printing callback with progress reporting and
    cooperative cancellation."""

    def attach_monitor(self, on_iteration, cancel_event):
        self._on_iteration = on_iteration
        self._cancel_event = cancel_event

    def callback(self, xr=None):
        if self._cancel_event.is_set():
            raise TrainingCancelled()
        lambd = None
        if hasattr(self.model, "lambd"):
            lambd = np.atleast_1d(self.model.lambd.numpy()).astype(float).tolist()
            self.model.lambd_list.append(self.model.lambd.numpy())
        self.hist.append(self.current_loss)
        self._on_iteration(self.iter, float(self.current_loss), lambd)
        self.iter += 1


class MonitoredPDESolver(_MonitorMixin, PINN_PDESolver):
    pass


class MonitoredIdentificationSolver(_MonitorMixin, PINN_IdentificationSolver):
    pass


class _ReportingList(list):
    """List that invokes ``on_append`` after every append."""

    def __init__(self, on_append):
        super().__init__()
        self._on_append = on_append

    def append(self, item):
        super().append(item)
        self._on_append(item)


class MonitoredHadamardNet(Hadamard_IdentificationNet):
    """The Hadamard solver's parameter phase runs its own inner callback
    (not ``solver.callback``), which records every step via
    ``model.lambd_list.append``. Swapping in a reporting list when the
    parameters are added lets us monitor that phase too."""

    on_param_step = None

    def add_param(self):
        super().add_param()
        self.lambd_list = _ReportingList(self.on_param_step)


class MonitoredHadamardSolver(_MonitorMixin, Hadamard_IdentificationSolver):
    def attach_monitor(self, on_iteration, cancel_event):
        super().attach_monitor(on_iteration, cancel_event)
        self.model.on_param_step = self._param_step

    def _param_step(self, lambd):
        # Called before the library's callback increments ``self.iter``.
        if self._cancel_event.is_set():
            raise TrainingCancelled()
        self._on_iteration(self.iter, float(self.current_loss),
                           np.atleast_1d(lambd).astype(float).tolist())

    def make_get_r_param(self):
        """Phase-2 loss and its gradient w.r.t. ``model.lambd`` (the
        ``get_r_param`` argument of ``solve_with_TFOptimizer``)."""

        def get_r_param():
            with tf.GradientTape() as tape:
                tape.watch(self.model.lambd)
                loss = tf.reduce_mean(tf.square(self.get_r()))
            return loss, tape.gradient(loss, self.model.lambd)

        return get_r_param


def _predict_grid(model, problem, nt=GRID_NT, nx=GRID_NX):
    """Evaluate the trained model and the exact solution on a regular grid.

    Returns (t, x, u_pred, u_exact) where the 2D arrays are indexed
    [x, t] (rows = space, columns = time), matching the frontend heatmaps.
    """
    tspace = np.linspace(problem.lb[0], problem.ub[0], nt + 1)
    xspace = np.linspace(problem.lb[1], problem.ub[1], nx + 1)
    T, X = np.meshgrid(tspace, xspace)
    Xgrid = np.vstack([T.flatten(), X.flatten()]).T

    u_pred = model(tf.cast(Xgrid, DTYPE)).numpy().reshape(nx + 1, nt + 1)
    u_exact = np.asarray(problem.u_exact(T, X), dtype=np.float64)
    return tspace, xspace, u_pred, u_exact


def _relative_l2_error(u_pred, u_exact):
    diff = (u_exact - u_pred).ravel()
    return float(np.linalg.norm(diff) / np.linalg.norm(u_exact.ravel()) * 100)


def run_training(request, on_iteration, cancel_event):
    """Train a PINN according to ``request`` (a ``TrainRequest``).

    ``on_iteration(iteration, loss, lambd_or_None)`` is invoked once per
    optimizer step. Raises ``TrainingCancelled`` if ``cancel_event`` is set.

    Returns a dict with the solution grids and metrics.
    """
    problem = get_problem(request.problem)
    tf.random.set_seed(request.seed)
    np.random.seed(request.seed)

    lb, ub = problem.lb_tensor, problem.ub_tensor
    arch = {
        "output_dim": 1,
        "num_hidden_layers": request.num_hidden_layers,
        "num_neurons_per_layer": request.num_neurons_per_layer,
    }

    t0 = time.time()

    if request.mode == "forward":
        X_data, u_data = generate_training_data(
            request.n_initial, request.n_boundary,
            problem.fun_u_0, problem.fun_u_b, lb, ub)
        X_r = generate_collocation_points(request.n_collocation, lb, ub)

        model = PINN_NeuralNet(lb, ub, **arch)
        model.build((None, 2))

        solver = MonitoredPDESolver(
            model, X_r, problem.make_fun_r(), problem.make_get_r())
    else:
        X_d, u_d = generate_training_data_inverse(
            request.n_data, lb, ub, problem.u_exact, noise=request.noise)
        u_d = tf.cast(u_d, DTYPE)
        X_data, u_data = [X_d], [u_d]
        X_r = generate_collocation_points(request.n_collocation, lb, ub)

        initial_lambda = request.initial_lambda
        if initial_lambda is None:
            initial_lambda = [0.0] * len(problem.lambda_star)
        if len(initial_lambda) != len(problem.lambda_star):
            raise ValueError(
                f"initial_lambda must have {len(problem.lambda_star)} "
                f"value(s) for problem '{problem.key}'")
        initial_lambda = np.array(initial_lambda, dtype=DTYPE)

        if request.mode == "inverse":
            model = PINN_IdentificationNet(
                initial_lambda=initial_lambda, lb=lb, ub=ub, **arch)
            solver_cls = MonitoredIdentificationSolver
        else:
            # ``model.lambd`` only exists once the solver calls add_param()
            # after the data-fitting phase; the residual reads it lazily.
            model = MonitoredHadamardNet(
                initial_lambda=initial_lambda, lb=lb, ub=ub, **arch)
            solver_cls = MonitoredHadamardSolver
        model.build((None, 2))

        solver = solver_cls(
            model, X_r, problem.make_fun_r_trainable(model),
            problem.make_get_r())

    solver.attach_monitor(on_iteration, cancel_event)

    def make_optimizer():
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            request.learning_rate, decay_steps=1000, decay_rate=0.9)
        return tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    if request.mode == "hadamard":
        solver.solve_with_TFOptimizer(
            make_optimizer(), make_optimizer(), solver.make_get_r_param(),
            X_data, u_data, N_fwd=request.epochs,
            N_param=request.param_epochs, modified=True)
    else:
        solver.solve_with_TFOptimizer(
            make_optimizer(), X_data, u_data, N=request.epochs)

    elapsed = time.time() - t0

    tspace, xspace, u_pred, u_exact = _predict_grid(model, problem)
    abs_error = np.abs(u_exact - u_pred)

    lambda_pred = None
    if hasattr(model, "lambd"):
        lambda_pred = np.atleast_1d(model.lambd.numpy()).astype(float).tolist()

    def _grid(a):
        return np.round(a, 5).tolist()

    return {
        "l2_error_percent": _relative_l2_error(u_pred, u_exact),
        "lambda_pred": lambda_pred,
        "lambda_star": problem.lambda_star,
        "lambda_names": problem.lambda_names,
        "elapsed_seconds": round(elapsed, 2),
        "solution": {
            "t": np.round(tspace, 5).tolist(),
            "x": np.round(xspace, 5).tolist(),
            "u_pred": _grid(u_pred),
            "u_exact": _grid(u_exact),
            "abs_error": _grid(abs_error),
        },
    }
