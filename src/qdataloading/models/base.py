import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
import os
from ..utils.metrics import evaluate

class BaseModel:
    def __init__(self, data_class, n_epoch: int, lr: float):
        self.data_class = data_class
        self.n_qubit = data_class.n_bit
        self.n_epoch = n_epoch
        self.lr = lr
        # JAX handles device management transparently, but we can print info if needed.
        self.target_prob = jnp.array(data_class.get_data())
        
        self.loss_history = []
        self.kl_history = []
        self.js_history = []
        self.grad_norm_history = []
        
        self.output_dir = "results"
        self.image_dir = "images"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)

    def fit(self):
        raise NotImplementedError

    def save_results(self, prob, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            data_dict = {
                'pmf': np.array(prob).tolist(),
                'kl div': self.kl_history,
                'js div': self.js_history,
                'loss history': self.loss_history,
                'grad norm': self.grad_norm_history
            }
            json.dump(data_dict, f)

    def plot_training_result(self, prob, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        fig = plt.figure(figsize=(10, 9))
        gs = gridspec.GridSpec(2, 1, figure=fig)
 
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])
        
        ax1_ = ax1.twinx()
        
        n_epoch = len(self.kl_history)
        epochs = np.arange(n_epoch) + 1

        ax1.semilogy(epochs, self.kl_history, label='KL divergence', color='red')
        ax1.semilogy(epochs, self.js_history, label='JS divergence', color='blue')
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('KL / JS divergence')
        
        ax1_.plot(epochs, self.loss_history, label='Loss', color='yellowgreen')
        ax1_.set_ylabel('Loss')
        
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles1_, labels1_ = ax1_.get_legend_handles_labels()
        ax1.legend(handles1 + handles1_, labels1 + labels1_, loc='upper right')
        ax1.grid()

        # Custom plotting logic for datasets
        from ..data.real_images import RealImage
        prob_np = np.array(prob)
        if isinstance(self.data_class, RealImage):
            # Try to infer shape, default to square
            dim = int(np.sqrt(len(prob_np)))
            if self.data_class.remapped:
                ax2.imshow(prob_np[self.data_class.inverse_indices].reshape(dim, dim))
            else:
                ax2.imshow(prob_np.reshape(dim, dim))
        else:
            ax2.bar(np.arange(prob_np.shape[0]) + 1, np.array(self.target_prob), alpha=0.5, color='blue', label='target')
            ax2.bar(np.arange(prob_np.shape[0]) + 1, prob_np, alpha=0.5, color='red', label='approx')
            ax2.legend(loc='upper right')

        plt.savefig(filename)
        plt.close()
