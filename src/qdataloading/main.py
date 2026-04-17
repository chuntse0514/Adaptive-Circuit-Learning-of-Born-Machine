import argparse
import yaml
import os
from qdataloading.data import get_dataset
from qdataloading.models.qcbm import QCBM
from qdataloading.models.aclbm import ACLBM

def main():
    parser = argparse.ArgumentParser(description="Quantum Data Loading Experiments")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print(f"Loading dataset: {config['dataset']}")
    dataset = get_dataset(config['dataset'])

    model_type = config['model'].lower()
    print(f"Initializing model: {model_type}")

    if model_type == 'qcbm':
        model = QCBM(
            data_class=dataset,
            n_epoch=config['n_epoch'],
            reps=config['reps'],
            lr=config['lr']
        )
    elif model_type == 'aclbm':
        model = ACLBM(
            data_class=dataset,
            n_epoch=config['n_epoch'],
            n_iter=config['n_iter'],
            No=config['No'],
            alpha=config['alpha'],
            reduction_rate=config.get('reduction_rate')
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    print("Starting training...")
    model.fit()
    print("Training finished.")

if __name__ == "__main__":
    main()
