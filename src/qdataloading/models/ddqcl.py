import jax
import jax.numpy as jnp
import optax
import numpy as np

from .base import BaseModel
from ..modules.ansatz import get_generator1, get_generator2
from ..modules.losses import kl_loss
from ..utils.metrics import evaluate

class DDQCL(BaseModel):
    def __init__(self, data_class, n_epoch: int, reps: int, lr: float):
        super().__init__(data_class, n_epoch, lr)
        self.reps = reps
        
        if data_class.dist_property == 'sparse':
            self.qnode, param_shape = get_generator1(self.n_qubit, reps)
        else:
            self.qnode, param_shape = get_generator2(self.n_qubit, reps)

        key = jax.random.PRNGKey(42)
        self.params = jax.random.normal(key, param_shape) * (jnp.pi / 8)
        
        self.optimizer = optax.adam(lr)
        self.opt_state = self.optimizer.init(self.params)

        self.filename = f'{self.image_dir}/DDQCL/DDQCL(data={data_class.name}, lr={lr}, reps={reps}).pdf'
        self.result_file = f'{self.output_dir}/DDQCL/DDQCL(data={data_class.name}, lr={lr}, reps={reps}).json'

    def fit(self, save=True):
        threshold = 1e-3
        
        @jax.jit
        def step(params, opt_state):
            def loss_fn(p):
                pred_prob = self.qnode(p, self.n_qubit, self.reps)
                return kl_loss(self.target_prob, pred_prob), pred_prob
            
            (loss, pred_prob), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            updates, opt_state = self.optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss, pred_prob, grads

        params = self.params
        opt_state = self.opt_state

        for i_epoch in range(self.n_epoch):
            params, opt_state, loss, pred_prob, grads = step(params, opt_state)
            grad_norm = jnp.linalg.norm(grads)
            
            self.loss_history.append(float(loss))
            kl_div, js_div = evaluate(self.target_prob, pred_prob)
            self.kl_history.append(kl_div)
            self.js_history.append(js_div)
            self.grad_norm_history.append(float(grad_norm))

            if (i_epoch + 1) % 100 == 0:
                print(f'epoch: {i_epoch+1} | loss: {float(loss):6f} | KL: {kl_div:6f} | JS: {js_div:6f} | Grad Norm: {float(grad_norm):6f}')

            if grad_norm < threshold:
                break
        
        if save:
            self.plot_training_result(pred_prob, self.filename)
            self.save_results(pred_prob, self.result_file)
        self.params = params
