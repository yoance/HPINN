"""In-memory training job manager.

Jobs run one at a time on a single worker thread (TensorFlow training is
CPU-bound; queueing keeps the demo responsive and predictable). State is
kept in memory — restarting the server clears jobs.
"""

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from . import training


@dataclass
class Job:
    job_id: str
    request: object
    status: str = "queued"  # queued | running | completed | failed | cancelled
    iteration: int = 0
    loss_history: List[float] = field(default_factory=list)
    lambda_history: List[List[float]] = field(default_factory=list)
    lambda_iterations: List[int] = field(default_factory=list)
    result: Optional[dict] = None
    error: Optional[str] = None
    cancel_event: threading.Event = field(default_factory=threading.Event)


class JobManager:
    def __init__(self):
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1)

    def submit(self, request) -> Job:
        job = Job(job_id=uuid.uuid4().hex[:12], request=request)
        with self._lock:
            self._jobs[job.job_id] = job
        self._executor.submit(self._run, job)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> Optional[Job]:
        job = self.get(job_id)
        if job and job.status in ("queued", "running"):
            job.cancel_event.set()
        return job

    def _run(self, job: Job):
        if job.cancel_event.is_set():
            job.status = "cancelled"
            return
        job.status = "running"

        def on_iteration(iteration, loss, lambd):
            job.iteration = iteration
            job.loss_history.append(loss)
            if lambd is not None:
                # iterations first: the API indexes them by len(lambda_history)
                job.lambda_iterations.append(iteration)
                job.lambda_history.append(lambd)

        try:
            job.result = training.run_training(
                job.request, on_iteration, job.cancel_event)
            job.status = "completed"
        except training.TrainingCancelled:
            job.status = "cancelled"
        except Exception as exc:  # surfaced to the client via job status
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {exc}"


manager = JobManager()
