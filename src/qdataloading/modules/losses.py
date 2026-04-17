import jax.numpy as jnp
import jax

def make_K(n_qubit, sigmas):
    sigmas = jnp.array(sigmas)
    x_range = jnp.arange(2 ** n_qubit)
    # distance matrix
    dist_sq = jnp.abs(x_range[:, None] - x_range[None, :]) ** 2
    
    # K_{ij} = sum_s exp(-|i-j|^2 / (2 * sigma_s^2)) / len(sigmas)
    K = jnp.mean(jnp.exp(-dist_sq[..., None] / (2 * sigmas ** 2)), axis=-1)
    return K

def mmd_loss(x, y, K, log=False):
    # x: target_prob, y: pred_prob
    # MMD^2 = (x-y)^T * K * (x-y)
    diff = x - y
    loss = jnp.dot(jnp.dot(diff, K), diff)
    
    if log:
        return jnp.log2(jnp.maximum(loss, 1e-18))
    
    return loss
