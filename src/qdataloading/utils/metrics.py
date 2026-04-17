import jax.numpy as jnp

def evaluate(target: jnp.ndarray, outcome: jnp.ndarray):
    assert target.shape == outcome.shape
    
    def kl_div_fn(p, q):
        # Filter where p > 0 and q > 0 to avoid NaN/Inf
        mask = (p > 0) & (q > 0)
        return jnp.sum(p[mask] * jnp.log(p[mask] / q[mask]))
        
    def js_div_fn(p, q):
        m = 0.5 * (p + q)
        return 0.5 * kl_div_fn(p, m) + 0.5 * kl_div_fn(q, m)
    
    return float(kl_div_fn(target, outcome)), float(js_div_fn(target, outcome))
