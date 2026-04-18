import jax
import jax.numpy as jnp
import optax
import numpy as np

from .base import BaseModel
from ..modules.ansatz import get_generator1, get_generator2
from ..utils.metrics import evaluate

def leaky_relu(x, negative_slope=0.01):
    return jnp.where(x > 0, x, x * negative_slope)

def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))

def discriminator_fn(params, x):
    # Manual MLP implementation
    # x shape (batch, 1)
    for w, b in params:
        x = jnp.dot(x, w) + b
        if w is not params[-1][0]: # not the last layer
            x = leaky_relu(x)
    return sigmoid(x)

class QGAN(BaseModel):
    def __init__(self, data_class, n_epoch: int, reps: int, lr: float):
        super().__init__(data_class, n_epoch, lr)
        self.reps = reps
        
        # Generator init
        if data_class.dist_property == 'sparse':
            self.qnode, g_param_shape = get_generator1(self.n_qubit, reps)
        else:
            self.qnode, g_param_shape = get_generator2(self.n_qubit, reps)

        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)
        self.g_params = jax.random.normal(k1, g_param_shape) * (jnp.pi / 8)
        
        # Discriminator init
        # Layer sizes: 1 -> 50 -> 20 -> 1
        d_layers = [1, 50, 20, 1]
        self.d_params = []
        for i in range(len(d_layers)-1):
            k2, sk = jax.random.split(k2)
            w = jax.random.normal(sk, (d_layers[i], d_layers[i+1])) * jnp.sqrt(2/d_layers[i])
            b = jnp.zeros(d_layers[i+1])
            self.d_params.append((w, b))

        # Optimizers
        self.g_optimizer = optax.adam(lr)
        self.g_opt_state = self.g_optimizer.init(self.g_params)
        
        self.d_optimizer = optax.adam(lr)
        self.d_opt_state = self.d_optimizer.init(self.d_params)

        self.loss_history = {'g loss': [], 'd loss': []}
        self.filename = f'{self.image_dir}/QGAN/QGAN(data={data_class.name}, lr={lr}, reps={reps}).pdf'
        self.result_file = f'{self.output_dir}/QGAN/QGAN(data={data_class.name}, lr={lr}, reps={reps}).json'

    def fit(self, save=True):
        best_kl = float('inf')
        best_pmf = None
        
        inputs = jnp.arange(2**self.n_qubit).reshape(-1, 1).astype(jnp.float32)
        epsilon = 1e-12

        @jax.jit
        def d_step(g_params, d_params, d_opt_state):
            gen_prob = self.qnode(g_params, self.n_qubit, self.reps)
            
            def loss_fn(dp):
                disc_prob = discriminator_fn(dp, inputs).squeeze()
                disc_prob = disc_prob / (jnp.linalg.norm(disc_prob) + epsilon)
                
                real_loss = -jnp.dot(self.target_prob, jnp.log(disc_prob + epsilon))
                fake_loss = -jnp.dot(gen_prob, jnp.log(1.0 - disc_prob + epsilon))
                return (real_loss + fake_loss) / 2
            
            loss, grads = jax.value_and_grad(loss_fn)(d_params)
            updates, d_opt_state = self.d_optimizer.update(grads, d_opt_state)
            d_params = optax.apply_updates(d_params, updates)
            return d_params, d_opt_state, loss

        @jax.jit
        def g_step(g_params, d_params, g_opt_state):
            disc_prob = discriminator_fn(d_params, inputs).squeeze()
            disc_prob = disc_prob / (jnp.linalg.norm(disc_prob) + epsilon)
            
            def loss_fn(gp):
                gen_prob = self.qnode(gp, self.n_qubit, self.reps)
                return -jnp.dot(gen_prob, jnp.log(disc_prob + epsilon))
            
            loss, grads = jax.value_and_grad(loss_fn)(g_params)
            updates, g_opt_state = self.g_optimizer.update(grads, g_opt_state)
            g_params = optax.apply_updates(g_params, updates)
            return g_params, g_opt_state, loss

        g_params, d_params = self.g_params, self.d_params
        g_opt_state, d_opt_state = self.g_opt_state, self.d_opt_state

        for i_epoch in range(self.n_epoch):
            d_params, d_opt_state, d_loss = d_step(g_params, d_params, d_opt_state)
            g_params, g_opt_state, g_loss = g_step(g_params, d_params, g_opt_state)
            
            prob = self.qnode(g_params, self.n_qubit, self.reps)
            kl_div, js_div = evaluate(self.target_prob, prob)
            
            self.kl_history.append(kl_div)
            self.js_history.append(js_div)
            self.loss_history['g loss'].append(float(g_loss))
            self.loss_history['d loss'].append(float(d_loss))
            
            if kl_div < best_kl:
                best_kl = kl_div
                best_pmf = prob
            
            if (i_epoch + 1) % 100 == 0:
                print(f'epoch: {i_epoch+1} | G_loss: {float(g_loss):6f} | D_loss: {float(d_loss):6f} | KL: {kl_div:6f}')

        if save:
            self.plot_training_result(best_pmf, self.filename)
            self.save_results(best_pmf, self.result_file)
        self.g_params, self.d_params = g_params, d_params
