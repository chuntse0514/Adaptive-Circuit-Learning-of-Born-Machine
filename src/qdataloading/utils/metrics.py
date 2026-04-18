import jax.numpy as jnp

def evaluate(target: jnp.ndarray, outcome: jnp.ndarray):
    assert target.shape == outcome.shape
    
    def kl_div_fn(p, q):
        # Functional way to avoid boolean indexing on tracers
        return jnp.sum(p * jnp.log(jnp.clip(p / jnp.clip(q, 1e-18, 1.0), 1e-18, 1e18)))
        
    def js_div_fn(p, q):
        m = 0.5 * (p + q)
        return 0.5 * kl_div_fn(p, m) + 0.5 * kl_div_fn(q, m)
    
    return float(kl_div_fn(target, outcome)), float(js_div_fn(target, outcome))
