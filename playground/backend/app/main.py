"""H-PINN FastAPI application.

Run from the ``playground/`` directory with (see ``playground/README.md``):

    uv run uvicorn backend.app.main:app --reload

The interactive frontend is served at http://localhost:8000/ and the API
docs at http://localhost:8000/docs.
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import schemas
from .jobs import manager
from hpinn import PROBLEMS

app = FastAPI(
    title="H-PINN API",
    description="Train (Hadamard) Physics-Informed Neural Networks on the "
                "heat and KdV equations — forward solving and inverse "
                "parameter identification (joint PINN or two-phase "
                "Hadamard-PINN).",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_HISTORY_POINTS = 400


def _decimate(seq, max_points=MAX_HISTORY_POINTS):
    """Downsample a history list for transport, always keeping the last point."""
    n = len(seq)
    if n <= max_points:
        return list(seq)
    step = n / max_points
    idx = [int(i * step) for i in range(max_points)]
    if idx[-1] != n - 1:
        idx.append(n - 1)
    return [seq[i] for i in idx]


@app.get("/api/problems", response_model=list[schemas.ProblemInfo])
def list_problems():
    return [
        schemas.ProblemInfo(
            key=p.key, name=p.name, description=p.description,
            lb=p.lb, ub=p.ub,
            lambda_star=p.lambda_star, lambda_names=p.lambda_names,
        )
        for p in PROBLEMS.values()
    ]


@app.post("/api/train", response_model=schemas.JobCreated, status_code=202)
def start_training(request: schemas.TrainRequest):
    if request.mode != "forward" and request.initial_lambda is not None:
        expected = len(PROBLEMS[request.problem].lambda_star)
        if len(request.initial_lambda) != expected:
            raise HTTPException(
                422,
                f"initial_lambda must have {expected} value(s) "
                f"for problem '{request.problem}'")
    job = manager.submit(request)
    return schemas.JobCreated(job_id=job.job_id)


@app.get("/api/train/{job_id}", response_model=schemas.JobStatus)
def job_status(job_id: str):
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return schemas.JobStatus(
        job_id=job.job_id,
        status=job.status,
        request=job.request,
        progress=schemas.JobProgress(
            iteration=job.iteration,
            total=job.request.total_epochs,
            loss_history=_decimate(job.loss_history),
            lambda_history=_decimate(job.lambda_history),
        ),
        error=job.error,
        result=job.result,
    )


@app.delete("/api/train/{job_id}", response_model=schemas.JobStatus)
def cancel_job(job_id: str):
    job = manager.cancel(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return job_status(job_id)


# Serve the demo frontend (must be mounted last so /api keeps precedence).
_FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
if _FRONTEND.exists():
    app.mount("/", StaticFiles(directory=_FRONTEND, html=True), name="frontend")
