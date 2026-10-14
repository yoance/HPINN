# H-PINN — Hadamard Physics-Informed Neural Networks

Physics-Informed Neural Networks (PINNs) for solving partial differential
equations and identifying their parameters, with a FastAPI backend and an
interactive web playground.

This is the code accompanying the published paper:

> Yohan Chandrasukmana, Helena Margaretha, Kie Van Ivanky Saputra,
> **The Hadamard-PINN for PDE inverse problems: Convergence with distant
> initial guesses**, *Examples and Counterexamples*, Volume 7, 2025, 100175.
> [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2666657X25000023) ·
> [doi:10.1016/j.exco.2025.100175](https://doi.org/10.1016/j.exco.2025.100175)

H-PINN (Hadamard-PINN) applies Hadamard's conditions for well-posed problems to the PDE
inverse problem: training is split into two phases — first the network is
fitted to the observed data alone, then the PDE parameters λ are searched
while the fitted solution is held as the anchor. This makes parameter
identification converge even from initial guesses far from the true values.
The method is implemented in `Hadamard_IdentificationNet` /
`Hadamard_IdentificationSolver` in `src/hpinn/pinn_pde_solver.py`, alongside
the standard PINN forward solver and joint inverse solver.

Supported problems:

| Problem | PDE residual | Domain | True λ |
|---|---|---|---|
| Heat equation | `u_t + λ u_xx` | t ∈ [0, 10], x ∈ [0, 6] | λ = −1 |
| KdV equation (soliton) | `u_t + λ₁ u u_x + λ₂ u_xxx` | t ∈ [−1, 1], x ∈ [−10, 10] | λ = [−6, 1] |

Both support two modes:

- **Forward** — learn u(t, x) from the physics residual plus initial and
  boundary conditions.
- **Inverse** — given (optionally noisy) measurements of u, identify the
  unknown PDE coefficients λ while fitting the solution.

## Project structure

```
├── src/hpinn/            Core library
│   ├── pinn_pde_solver.py   PINN models & solvers (TF/Keras 2 API)
│   ├── problems.py          Heat & KdV problem definitions
│   └── combined_decay.py    CombinedDecay learning-rate schedule
└── notebooks/            Colab notebooks — proof-of-concept
```


## Installation

Python 3.10+.

**As a library** — install straight from GitHub and `import hpinn` in your
own projects:

```bash
pip install "git+https://github.com/yoance/HPINN.git"
```

```python
import hpinn
print(hpinn.__version__)
```


## Citation

If you use this code or the H-PINN method in your research, please cite:

```bibtex
@article{chandrasukmana2025hadamard,
  title   = {The Hadamard-PINN for PDE inverse problems: Convergence with distant initial guesses},
  author  = {Chandrasukmana, Yohan and Margaretha, Helena and Saputra, Kie Van Ivanky},
  journal = {Examples and Counterexamples},
  volume  = {7},
  pages   = {100175},
  year    = {2025},
  doi     = {10.1016/j.exco.2025.100175},
}
```

## Note on TensorFlow versions

The solvers use the Keras 2 API. On TensorFlow ≥ 2.16 (which ships Keras 3)
the code transparently falls back to the official
[`tf-keras`](https://pypi.org/project/tf-keras/) package via
`TF_USE_LEGACY_KERAS=1`, which is set automatically when `hpinn` is imported
before TensorFlow.
