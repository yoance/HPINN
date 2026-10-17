# H-PINN: Hadamard Physics-Informed Neural Networks

Physics-informed neural networks (PINNs) for solving partial differential
equations and identifying their parameters.

This repository contains the code for the paper:

> Yohan Chandrasukmana, Helena Margaretha, Kie Van Ivanky Saputra,
> **The Hadamard-PINN for PDE inverse problems: Convergence with distant
> initial guesses**, *Examples and Counterexamples*, Volume 7, 2025, 100175.
> [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2666657X25000023) ·
> [doi:10.1016/j.exco.2025.100175](https://doi.org/10.1016/j.exco.2025.100175)

H-PINN (Hadamard-PINN) applies Hadamard's conditions for well-posed problems
to the PDE inverse problem. Training runs in two phases. First the network is
fitted to the observed data alone. Then the PDE parameters λ are searched
while the fitted solution is held as the anchor. With this split, parameter
identification converges even when the initial guesses are far from the true
values.

The method is in `Hadamard_IdentificationNet` and
`Hadamard_IdentificationSolver` in `src/hpinn/pinn_pde_solver.py`, next to the
standard PINN forward solver and the joint inverse solver.

Two problems are implemented:

| Problem | PDE residual | Domain | True λ |
|---|---|---|---|
| Heat equation | `u_t + λ u_xx` | t ∈ [0, 10], x ∈ [0, 6] | λ = −1 |
| KdV equation (soliton) | `u_t + λ₁ u u_x + λ₂ u_xxx` | t ∈ [−1, 1], x ∈ [−10, 10] | λ = [−6, 1] |

Each one runs in two modes. 
1. In **forward mode**, the network learns u(t, x) from the physics 
residual plus the initial and boundary conditions.
2. In **inverse mode**, it takes measurements of u (optionally noisy) 
and identifies the unknown PDE coefficients λ while fitting the solution.

## Project structure

```
├── src/hpinn/            Core library
│   ├── pinn_pde_solver.py   PINN models & solvers (TF/Keras 2 API)
│   ├── problems.py          Heat & KdV problem definitions
│   └── schedulers/          CombinedDecay learning-rate schedule
├── notebooks/            Colab notebooks — proof-of-concept
└── playground/           Interactive web demo (see playground/README.md)
```


## Installation

Requires Python 3.10+.

### As a library

Install from GitHub, then `import hpinn` in your own projects:

```bash
pip install "git+https://github.com/yoance/HPINN.git"
```

```python
import hpinn
print(hpinn.__version__)
```

### For development

The repository is managed with [uv](https://docs.astral.sh/uv/). Clone it and
sync the environment. uv takes the Python version from `.python-version` and
the pinned dependencies from `uv.lock`:

```bash
git clone https://github.com/yoance/HPINN.git
cd HPINN
uv sync
```

Add `--extra notebooks` to install the Jupyter notebook dependencies. The web
playground has its own dependency group and setup; see
[`playground/README.md`](playground/README.md).


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

The solvers use the Keras 2 API. TensorFlow 2.16 and later ship Keras 3, so on
those versions the code falls back to the official
[`tf-keras`](https://pypi.org/project/tf-keras/) package by setting
`TF_USE_LEGACY_KERAS=1`. `hpinn` sets this variable automatically as long as
you import it before TensorFlow.