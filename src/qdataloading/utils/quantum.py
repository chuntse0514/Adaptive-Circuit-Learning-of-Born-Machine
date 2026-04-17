import jax.numpy as jnp
import numpy as np

def partial_trace(state: jnp.ndarray, dims: list[int]):
    n_qubit = int(np.log2(state.shape[0]))
    rdm_dim = 2 ** (n_qubit - len(dims))

    # Reshape to (2, 2, ..., 2)
    state = state.reshape(*[2 for _ in range(n_qubit)])
    
    # JAX equivalent of torch.tensordot
    rdm = jnp.tensordot(state, jnp.conjugate(state), axes=(dims, dims))
    return rdm.reshape(rdm_dim, rdm_dim)


def von_Neumann_entropy(state: jnp.ndarray):
    # state is a density matrix
    eigvals = jnp.linalg.eigvalsh(state).real
    eigvals = jnp.maximum(eigvals, 0.0)
    # Filter out zeros to avoid log(0)
    mask = eigvals > 0
    return -jnp.sum(eigvals[mask] * jnp.log2(eigvals[mask]))


def mutual_information(state: jnp.ndarray, subsystems: tuple):
    # state is a state vector (amplitude embedding)
    # First, convert to density matrix if it's a state vector
    # But partial_trace above already takes the state vector and dots it with conjugate.
    # Wait, the partial_trace implementation I wrote above assumes 'state' is the amplitude vector.
    # Let's verify if 'state' passed is amplitude or density matrix.
    # In the original code, state = torch.sqrt(target_prob), which is amplitudes (real-valued for BM).
    
    # convert to the data structure tuple[list, list]
    subsystems = tuple([subsystem] if isinstance(subsystem, int) else list(subsystem) for subsystem in subsystems)
    entire_system = list(subsystems[0]) + list(subsystems[1])

    n_qubit = int(np.log2(state.shape[0]))
    
    rho_AB = partial_trace(state, dims=[i for i in range(n_qubit) if i not in entire_system])
    rho_A = partial_trace(state, dims=[i for i in range(n_qubit) if i not in subsystems[0]])
    rho_B = partial_trace(state, dims=[i for i in range(n_qubit) if i not in subsystems[1]])

    return float(von_Neumann_entropy(rho_A) + von_Neumann_entropy(rho_B) - von_Neumann_entropy(rho_AB))


def concurrence(rho: jnp.ndarray):
    # Only for 2-qubit states (4x4 rho)
    # This is more complex in JAX because of the flipping.
    # sigma_y = [[0, -1j], [1j, 0]]
    # sigma_y_y = sigma_y \otimes sigma_y
    
    sigma_y = jnp.array([[0, -1j], [1j, 0]])
    sigma_y_y = jnp.kron(sigma_y, sigma_y)
    
    rho_tilde = jnp.matmul(jnp.matmul(sigma_y_y, jnp.conjugate(rho)), sigma_y_y)
    
    # R = sqrt(sqrt(rho) * rho_tilde * sqrt(rho))
    # Or more simply, eigenvalues of rho * rho_tilde
    evals = jnp.linalg.eigvals(jnp.matmul(rho, rho_tilde))
    # evals are \lambda_i^2. We need \lambda_i
    lambdas = jnp.sort(jnp.sqrt(jnp.maximum(evals.real, 0.0)))[::-1]
    
    return float(jnp.maximum(0, lambdas[0] - jnp.sum(lambdas[1:])))


def entanglement_of_formation(state: jnp.ndarray, subsystems: tuple[int, int]):
    n_qubit = int(np.log2(state.shape[0]))

    rho = partial_trace(state, dims=[i for i in range(n_qubit) if i not in subsystems])
    C = concurrence(rho)
    
    # Avoid NaN in sqrt
    C = jnp.clip(C, 0.0, 1.0)
    
    def h(x):
        # binary entropy function
        return -x * jnp.log2(x) - (1 - x) * jnp.log2(1 - x)

    x = (1 + jnp.sqrt(1 - C ** 2)) / 2
    # Avoid log(0)
    x = jnp.clip(x, 1e-15, 1 - 1e-15)
    
    return float(h(x))
