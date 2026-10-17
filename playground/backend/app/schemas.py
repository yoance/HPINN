"""Pydantic request/response models for the H-PINN API."""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class TrainRequest(BaseModel):
    problem: Literal["heat", "kdv"] = "heat"
    mode: Literal["forward", "inverse"] = "forward"
    # Inverse mode only: "pinn" fits the network and λ jointly; "hpinn"
    # (Hadamard-PINN) first fits the network to the data for ``epochs``, then
    # searches λ for ``param_epochs`` with the fitted network held fixed.
    method: Literal["pinn", "hpinn"] = "pinn"

    # Network architecture
    num_hidden_layers: int = Field(2, ge=1, le=8)
    num_neurons_per_layer: int = Field(20, ge=5, le=100)

    # Training
    epochs: int = Field(2000, ge=10, le=20000)
    param_epochs: int = Field(1000, ge=10, le=20000,
                              description="λ-search epochs (hpinn method)")
    seed: int = 0

    # Learning-rate schedule (each optimizer starts its own schedule, so with
    # hpinn the schedule restarts for the λ-search phase)
    learning_rate: float = Field(1e-2, gt=0, le=1.0,
                                 description="Initial learning rate")
    lr_schedule: Literal["constant", "exponential", "combined"] = "exponential"
    decay_steps: int = Field(1000, ge=1, le=100000,
                             description="Exponential decay: lr · rate^(step / decay_steps)")
    decay_rate: float = Field(0.9, gt=0, le=1.0)
    boundaries: List[int] = Field(
        [1000, 1500], min_length=1, max_length=10,
        description="CombinedDecay: steps where the exponential decay hands "
                    "over to piecewise-constant values")
    values_steps: float = Field(
        2.0, ge=1.0, le=100.0,
        description="CombinedDecay: divide the learning rate by this at each boundary")

    # Forward problem data sizes
    n_initial: int = Field(50, ge=10, le=2000, description="N_0 initial condition points")
    n_boundary: int = Field(50, ge=10, le=2000, description="N_b boundary points")
    n_collocation: int = Field(1000, ge=100, le=20000, description="N_r collocation points")

    # Inverse problem settings
    n_data: int = Field(500, ge=50, le=5000, description="N_d measurement points")
    noise: float = Field(0.0, ge=0.0, le=1.0)
    initial_lambda: Optional[List[float]] = Field(
        None, description="Initial guess for the PDE parameters")

    @field_validator("boundaries")
    @classmethod
    def _increasing(cls, v):
        if v[0] < 1 or any(b <= a for a, b in zip(v, v[1:])):
            raise ValueError("boundaries must be positive and strictly increasing")
        return v

    @property
    def is_hpinn(self) -> bool:
        return self.mode == "inverse" and self.method == "hpinn"

    @property
    def total_epochs(self) -> int:
        return self.epochs + (self.param_epochs if self.is_hpinn else 0)


class JobProgress(BaseModel):
    iteration: int = 0
    total: int
    loss_history: List[float] = []
    loss_iterations: List[int] = []    # iteration of each loss_history point
    lambda_history: List[List[float]] = []
    lambda_iterations: List[int] = []  # iteration of each lambda_history point


class SolutionGrid(BaseModel):
    t: List[float]
    x: List[float]
    u_pred: List[List[float]]
    u_exact: List[List[float]]
    abs_error: List[List[float]]


class JobResult(BaseModel):
    l2_error_percent: float
    lambda_pred: Optional[List[float]] = None
    lambda_star: List[float]
    lambda_names: List[str]
    elapsed_seconds: float
    solution: SolutionGrid


class JobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    request: TrainRequest
    progress: JobProgress
    error: Optional[str] = None
    result: Optional[JobResult] = None


class JobCreated(BaseModel):
    job_id: str


class ProblemInfo(BaseModel):
    key: str
    name: str
    description: str
    lb: List[float]
    ub: List[float]
    lambda_star: List[float]
    lambda_names: List[str]
