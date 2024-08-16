import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
import torch

epsilon = 1e-12


def bits_to_ints(bits: np.array, n_bit: int) -> np.array:
    bits = bits.astype(int)
    base = np.reshape(2 ** np.arange(0, n_bit)[::-1], [n_bit, 1]).astype(int)
    ints = (bits @ base)[:, 0]
    return ints


def ints_to_bits(ints: np.array, n_bit: int) -> np.array:
    bits = np.zeros([len(ints), n_bit])
    template = f'{{0:0{n_bit}b}}'
    for idx, i in enumerate(ints):
        s = template.format(i)
        for b in range(n_bit):
            bits[idx][b] = int(s[b])
    return bits

def ints_to_onehot(ints: np.array, num_class: int) -> np.array:
    ret = np.zeros([len(ints), num_class])
    for idx, i in enumerate(ints):
        ret[idx][i] = 1
    return ret

def evaluate(target: torch.Tensor, outcome: torch.Tensor):
    
    assert(target.shape == outcome.shape)
    
    kl_div_fn = lambda p, q: torch.inner(p[p>0], torch.log(p[p>0] / q[p>0]))
    js_div_fn = lambda p, q: 1 / 2 * kl_div_fn(p, (p+q)/2) + 1 / 2 * kl_div_fn(q, (p+q)/2)
    
    return kl_div_fn(target, outcome).item(), js_div_fn(target, outcome).item()

def partial_trace(state: torch.Tensor, dims=list[int]):
    
    n_qubit = int(np.log2(state.shape[0]))
    rdm_dim = 2 ** (n_qubit - len(dims))

    state = state.view(*[2 for _ in range(n_qubit)])
    rdm = torch.tensordot(state, state.conj(), dims=(dims,dims))
    return rdm.view(rdm_dim, rdm_dim)

def von_Neumann_entropy(state: torch.Tensor):

    # return the non-zero eigenvalues
    eigvals = torch.linalg.eigvals(state).real
    eigvals = torch.maximum(eigvals, torch.zeros_like(eigvals))
    return -torch.inner(eigvals, torch.log2(eigvals)).item()

def mutual_information(state: torch.Tensor, subsystems: tuple[list | int, list | int]):

    # convert to the data structure tuple[list, list]
    subsystems = tuple([subsystem] for subsystem in subsystems if isinstance(subsystem, int))
    entire_system = subsystems[0] + subsystems[1]

    n_qubit = int(np.log2(state.shape[0]))
    
    rho_AB = partial_trace(state, dims=[i for i in range(n_qubit) if i not in entire_system])
    rho_A = partial_trace(state, dims=[i for i in range(n_qubit) if i not in subsystems[0]])
    rho_B = partial_trace(state, dims=[i for i in range(n_qubit) if i not in subsystems[1]])

    return von_Neumann_entropy(rho_A) + von_Neumann_entropy(rho_B) - von_Neumann_entropy(rho_AB)

def concurrence(rho: torch.Tensor):

    rho_eigvals, rho_eigvecs = torch.linalg.eigh(rho)
    rho_sqrt = rho_eigvecs @ torch.diag(torch.sqrt(rho_eigvals)) @ rho_eigvecs.conj().T

    sigma_YY = torch.fliplr(torch.diag(torch.Tensor([-1, 1, 1, -1]))).double().to(rho.device)
    rho_tilde = sigma_YY @ rho.conj() @ sigma_YY

    R_eigvals = torch.sqrt(torch.linalg.eigvals(rho_sqrt @ rho_tilde @ rho_sqrt).real)

    return max(0, (R_eigvals[0] - torch.sum(R_eigvals[1:])).item())

def entanglement_of_formation(state: torch.Tensor, subsystems=tuple[int, int]):

    n_qubit = int(np.log2(state.shape[0]))

    rho = partial_trace(state, dims=[i for i in range(n_qubit) if i not in subsystems])
    C = concurrence(rho)
    if C == 0:
        return 0
    x = (1 + np.sqrt(1 - C ** 2)) / 2
    return -x * np.log2(x) - (1 - x) * np.log2(1 - x)