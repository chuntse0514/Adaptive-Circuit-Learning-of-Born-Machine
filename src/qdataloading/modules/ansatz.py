import pennylane as qml
import jax.numpy as jnp
from pennylane import numpy as np

def generator1_circuit(params, n_qubit, k, pairs=None):
    # params is a list/array of parameters for each layer
    # shape: (k+1, n_qubit * 3)
    
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
    
    # Final layer
    for q in range(n_qubit):
        qml.RX(params[-1, 3*q], wires=q)
        qml.RY(params[-1, 3*q+1], wires=q)
        qml.RX(params[-1, 3*q+2], wires=q)
    
    return qml.probs(wires=range(n_qubit))

def generator2_circuit(params, n_qubit, k, pairs=None):
    # params shape: (k, n_qubit * 2)
    
    for layer in range(k):
        for q in range(n_qubit):
            qml.RY(params[layer, q], wires=q)
        if pairs:
            for i, j in pairs:
                qml.CRZ(params[layer, n_qubit+i], wires=[i, j]) # using i as an index for CRZ param
        else:
            for q in range(n_qubit):
                qml.CRZ(params[layer, n_qubit+q], wires=[q, (q+1) % n_qubit])
                
    return qml.probs(wires=range(n_qubit))

def get_generator1(n_qubit, k, pairs=None):
    dev = qml.device('default.qubit', wires=n_qubit)
    return qml.QNode(generator1_circuit, dev, interface='jax'), (k+1, n_qubit * 3)

def get_generator2(n_qubit, k, pairs=None):
    dev = qml.device('default.qubit', wires=n_qubit)
    return qml.QNode(generator2_circuit, dev, interface='jax'), (k, n_qubit * 2)
