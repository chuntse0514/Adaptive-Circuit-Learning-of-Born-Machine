import jax.numpy as jnp
import jax

def make_K(n_qubit, sigmas):
    sigmas = jnp.array(sigmas)
    x_range = jnp.arange(2 ** n_qubit)
    dist_sq = jnp.abs(x_range[:, None] - x_range[None, :]) ** 2
    K = jnp.mean(jnp.exp(-dist_sq[..., None] / (2 * sigmas ** 2)), axis=-1)
    return K

def mmd_loss(x, y, K, log=False):
    diff = x - y
    loss = jnp.dot(jnp.dot(diff, K), diff)
    # Use jnp.where for tracing safety
    if log:
        return jnp.log2(jnp.maximum(loss, 1e-18))
    return loss

def kl_loss(p, q):
    # Safe KL for tracing
    return jnp.sum(p * jnp.log(jnp.clip(p / jnp.clip(q, 1e-18, 1.0), 1e-18, 1e18)))
