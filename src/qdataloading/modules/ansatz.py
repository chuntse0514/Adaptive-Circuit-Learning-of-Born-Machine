import pennylane as qml
import jax.numpy as jnp
from pennylane import numpy as np

def generator1_circuit(params, n_qubit, k, pairs=None):
    for layer in range(k):
        for q in range(n_qubit):
            qml.RX(params[layer, 3*q], wires=q)
            qml.RY(params[layer, 3*q+1], wires=q)
            qml.RX(params[layer, 3*q+2], wires=q)
        if pairs:
            for i, j in pairs:
                qml.CZ(wires=[i, j])
        else:
            for q in range(n_qubit):
                qml.CZ(wires=[q, (q+1) % n_qubit])
    for q in range(n_qubit):
        qml.RX(params[-1, 3*q], wires=q)
        qml.RY(params[-1, 3*q+1], wires=q)
        qml.RX(params[-1, 3*q+2], wires=q)
    return qml.probs(wires=range(n_qubit))

def generator2_circuit(params, n_qubit, k, pairs=None):
    for layer in range(k):
        for q in range(n_qubit):
            qml.RY(params[layer, q], wires=q)
        if pairs:
            for i, j in pairs:
                qml.CRZ(params[layer, n_qubit + (i % n_qubit)], wires=[i, j])
        else:
            for q in range(n_qubit):
                qml.CRZ(params[layer, n_qubit+q], wires=[q, (q+1) % n_qubit])
    return qml.probs(wires=range(n_qubit))

def mps_circuit(params, n_qubit, k, pool_gates):
    for layer in range(k):
        for q in range(n_qubit - 1):
            for i in range(15):
                # Now gate is partial(TwoLocalPauliRotation, ps)
                # It expects (theta, qubits)
                pool_gates[i](params[layer, q, i], [q, q+1])
    return qml.probs(wires=range(n_qubit))

def get_generator1(n_qubit, k, pairs=None):
    dev = qml.device('default.qubit', wires=n_qubit)
    return qml.QNode(generator1_circuit, dev, interface='jax'), (k+1, n_qubit * 3)

def get_generator2(n_qubit, k, pairs=None):
    dev = qml.device('default.qubit', wires=n_qubit)
    return qml.QNode(generator2_circuit, dev, interface='jax'), (k, n_qubit * 2)

def get_mps_generator(n_qubit, k):
    from .gates import TwoLocalPauliRotation
    from functools import partial
    
    pauli_strings = ['IX', 'IY', 'IZ', 'XI', 'XX', 'XY', 'XZ', 'YI', 'YX', 'YY', 'YZ', 'ZI', 'ZX', 'ZY', 'ZZ']
    # partial(TwoLocalPauliRotation, pauli_string=ps) would work too if we keep it as keyword, 
    # but positional is safer for avoiding clashes.
    pool_gates = [partial(TwoLocalPauliRotation, ps) for ps in pauli_strings]
    
    dev = qml.device('default.qubit', wires=n_qubit)
    return qml.QNode(mps_circuit, dev, interface='jax'), (k, n_qubit-1, 15), pool_gates
