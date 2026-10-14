# H-PINN Playground

An interactive web demo for the `hpinn` library: train PINNs on the heat and
KdV equations from the browser and watch the loss, the identified PDE
parameters λ and the predicted solution u(t, x) as training runs.

The playground imports the solvers and problem definitions from
[`src/hpinn`](../src/hpinn) — it holds no PDE or model code of its own.

## Structure

```
playground/
├── backend/app/          FastAPI service
│   ├── main.py              API routes + static frontend hosting
│   ├── training.py          Training service (progress, cancellation)
│   ├── jobs.py              In-memory background job manager
│   └── schemas.py           Pydantic request/response models
└── frontend/             Web UI (vanilla HTML/CSS/JS)
```

## Setup

The playground shares the root project's environment (`pyproject.toml`,
`.python-version`, `uv.lock`). Its extra dependencies (FastAPI, Uvicorn) live
in the `playground` dependency group, so they are not part of the core
`hpinn` install. From the repository root:

```bash
uv sync --group playground
```

## Run

From the `playground/` directory:

```bash
uv run uvicorn backend.app.main:app --reload
```

Then open http://localhost:8000/ for the playground and
http://localhost:8000/docs for the interactive API docs.

## Modes

| Mode | Solver | What it does |
|---|---|---|
| `forward` | `PINN_PDESolver` | Learns u(t, x) from the PDE residual plus initial/boundary conditions. |
| `inverse` | `PINN_IdentificationSolver` | Fits measurements of u and the PDE parameters λ jointly. |
| `hadamard` | `Hadamard_IdentificationSolver` | H-PINN: first fits the network to the measurements only (`epochs`), then searches λ with the fitted network held fixed (`param_epochs`). |

## API

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/problems` | Available problems, domains and true λ. |
| `POST` | `/api/train` | Start a training job; returns its `job_id`. |
| `GET` | `/api/train/{job_id}` | Job status, progress (loss/λ history) and result. |
| `DELETE` | `/api/train/{job_id}` | Cancel a queued or running job. |

Jobs run one at a time on a background thread and are kept in memory, so
restarting the server clears them.
