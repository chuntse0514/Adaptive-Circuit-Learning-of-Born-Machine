# Adaptive-Circuit-Learning-of-Born-Machine

A JAX/PennyLane implementation of our paper:
[Adaptive Circuit Learning of Born Machine: Towards Realization of Amplitude Embedding and Quantum Data Loading](https://arxiv.org/abs/2311.17798)

## Project Structure
- `src/qdataloading/`: Core package
  - `data/`: Data loading and datasets (lazy loading)
  - `models/`: JAX-based model implementations (ACLBM, QCBM, QGAN, etc.)
  - `modules/`: Shared components (ansatz, losses, gates)
  - `utils/`: JAX utilities for metrics and quantum operations
- `configs/`: YAML configuration files for experiments
- `results/`: Output JSON files
- `images/`: Generated plots and training results

## Installation
```bash
pip install .
```
Or install dependencies manually:
```bash
pip install pennylane jax jaxlib optax pyyaml matplotlib scipy
```

## Running Experiments
Use the central entry point `main.py` with a configuration file:
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
python src/qdataloading/main.py --config configs/qcbm_bas3x3.yaml
```
