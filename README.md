# Adaptive Circuit Learning of Born Machine (ACLBM)

Official implementation of the paper: **"Adaptive circuit learning of Born machine: towards realization of amplitude embedding and quantum data loading"**, published in *Quantum Science and Technology* (2025).

This repository provides a high-performance JAX-based framework for quantum data loading using Adaptive Circuit Learning. It supports various Born Machine architectures and provides automated tools for training on both synthetic and real-world image datasets.

## 🚀 Key Features

- **JAX & PennyLane Integration:** Fully migrated to JAX for automatic differentiation, JIT compilation, and seamless GPU/TPU acceleration.
- **Adaptive Architecture:** Implementation of the **ACLBM** algorithm, which dynamically grows the quantum circuit by selecting optimal gates from an operator pool based on gradient information.
- **Multi-Model Support:** Includes benchmarks and implementations for:
  - **ACLBM:** Adaptive Circuit Learning of Born Machine.
  - **QCBM:** Quantum Circuit Born Machine.
  - **QGAN:** Quantum Generative Adversarial Network.
  - **DDQCL:** Data-Driven Quantum Circuit Learning.
  - **MPS:** Matrix Product State inspired circuits.
- **Automated Data Loading:** Dynamic discovery of image datasets. Simply drop an image into the `images/` folder, and the system registers it automatically.
- **Reproducible Experiments:** Configuration-driven execution using YAML files.

## 📁 Project Structure

```text
/
├── configs/            # YAML configuration files for experiments
├── images/             # Generated plots and training results (.pdf)
├── results/            # Detailed training metrics and probability distributions (.json)
└── src/qdataloading/   # Core Python package
    ├── data/           # Dataset classes and automated image discovery
    ├── models/         # JAX implementations of Born Machine variants
    ├── modules/        # Shared components (ansatz, losses, gates)
    └── utils/          # Quantum utilities (metrics, information theory)
```

## 🛠️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/chuntseli/Adaptive-Circuit-Learning-of-Born-Machine.git
cd Adaptive-Circuit-Learning-of-Born-Machine
```

### 2. Install dependencies
Ensure you have a JAX-compatible environment. We recommend using `conda`.
```bash
pip install .
```
Or install manually:
```bash
pip install pennylane jax jaxlib optax pyyaml matplotlib scipy pillow
```

## 📈 Running Experiments

Experiments are managed via the central `main.py` entry point. Use the `--config` flag to point to a YAML file.

```bash
# Set PYTHONPATH to include the src directory
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

# Run a QCBM experiment on the Bar-and-Stripes dataset
python src/qdataloading/main.py --config configs/qcbm_bas3x3.yaml

# Run ACLBM on a real image
python src/qdataloading/main.py --config configs/aclbm_realimage1.yaml
```

### Configuration Example
```yaml
model: qcbm
dataset: bas 3x3
n_epoch: 1000
reps: 10
lr: 0.05
```

## 🖼️ Working with Images

The repository automatically registers any image placed in `src/qdataloading/data/images/`.
- **Standard Registration:** Registered as `name` (e.g., `real image 1`).
- **Remapped Registration:** Registered as `name (R)`, which sorts pixels by intensity to simplify the learning task for the Born Machine.

All training result figures are saved as high-quality **PDF** files in the `images/` directory.

## 📝 Citation

If you use this code or the ACLBM algorithm in your research, please cite our journal paper:

```bibtex
@article{Li2025adaptive,
  title={Adaptive circuit learning of Born machine: towards realization of amplitude embedding and quantum data loading},
  author={Li, Chun-Tse and Cheng, Hao-Chung},
  journal={Quantum Science and Technology},
  volume={10},
  number={2},
  pages={025019},
  year={2025},
  publisher={IOP Publishing},
  doi={10.1088/2058-9565/adaede}
}
```

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
