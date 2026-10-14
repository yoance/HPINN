"""H-PINN: (Hadamard) Physics-Informed Neural Networks for solving PDEs.

Core library containing the PINN models, solvers, learning-rate schedules
and the predefined PDE problems (heat equation, KdV equation).
"""

import os
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("hpinn")
except PackageNotFoundError:  # running from a source checkout
    __version__ = "0.1.0"

# The solvers are written against the Keras 2 API. With TensorFlow >= 2.16
# (Keras 3) this flag routes tf.keras to the tf-keras compatibility package.
# Must be set before tensorflow is imported.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

from .pinn_pde_solver import (
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
    predict_solution,
    evaluate_solution,
    evaluate_solution_log_time,
    evaluate_param,
)
from .schedulers import CombinedDecay
from .problems import PROBLEMS, PDEProblem, get_problem

__all__ = [
    "DTYPE",
    "PINN_NeuralNet",
    "PINN_PDESolver",
    "PINN_IdentificationNet",
    "PINN_IdentificationSolver",
    "Hadamard_IdentificationNet",
    "Hadamard_IdentificationSolver",
    "generate_training_data",
    "generate_collocation_points",
    "generate_training_data_inverse",
    "predict_solution",
    "evaluate_solution",
    "evaluate_solution_log_time",
    "evaluate_param",
    "CombinedDecay",
    "PROBLEMS",
    "PDEProblem",
    "get_problem",
]
