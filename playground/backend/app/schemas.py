"""Pydantic request/response models for the H-PINN API."""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class TrainRequest(BaseModel):
    problem: Literal["heat", "kdv"] = "heat"
    mode: Literal["forward", "inverse", "hadamard"] = "forward"

    # Network architecture
    num_hidden_layers: int = Field(2, ge=1, le=8)
    num_neurons_per_layer: int = Field(20, ge=5, le=100)

    # Training
    epochs: int = Field(2000, ge=10, le=20000)
    learning_rate: float = Field(1e-2, gt=0, le=1.0)
    seed: int = 0

    # Forward problem data sizes
    n_initial: int = Field(50, ge=10, le=2000, description="N_0 initial condition points")
    n_boundary: int = Field(50, ge=10, le=2000, description="N_b boundary points")
    n_collocation: int = Field(1000, ge=100, le=20000, description="N_r collocation points")

    # Inverse problem settings (inverse and hadamard modes)
    n_data: int = Field(500, ge=50, le=5000, description="N_d measurement points")
    noise: float = Field(0.0, ge=0.0, le=1.0)
    initial_lambda: Optional[List[float]] = Field(
        None, description="Initial guess for the PDE parameters")

    # Hadamard mode: ``epochs`` fits the network to the data, then
    # ``param_epochs`` searches the PDE parameters with the network held fixed.
    param_epochs: int = Field(1000, ge=10, le=20000,
                              description="Parameter-search epochs (hadamard mode)")

    @property
    def total_epochs(self) -> int:
        if self.mode == "hadamard":
            return self.epochs + self.param_epochs
        return self.epochs


class JobProgress(BaseModel):
    iteration: int = 0
    total: int
    loss_history: List[float] = []
    lambda_history: List[List[float]] = []


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
