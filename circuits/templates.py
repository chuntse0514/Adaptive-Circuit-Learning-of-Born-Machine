import pennylane as qml
import torch
import numpy as np

def entangle_layer(n_qubit: int):
    for q in range(n_qubit):
        qml.CZ(wires=[q, (q+1) % n_qubit])

def RY_rot_layer(params, n_qubit: int):
    for q in range(n_qubit):
        qml.RY(params[q], wires=q)

def SU2_rot_layer(params, n_qubit: int):
    for q in range(n_qubit):
        qml.RX(params[q, 0], wires=q)
        qml.RY(params[q, 1], wires=q)
        qml.RX(params[q, 2], wires=q)

def RYCircuitModel(n_qubit, circuit_depth):
    
    dev = qml.device("default.qubit.torch", wires=n_qubit)
    
    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(params, su2_params):

        SU2_rot_layer(su2_params, n_qubit)
        for l in range(circuit_depth):
            entangle_layer(n_qubit)
            RY_rot_layer(params[l], n_qubit)
        
        return qml.probs(wires=list(range(n_qubit)))
    
    return circuit

def MølmerSørensenXXLayer(params, n_qubit: int):
    for q in range(n_qubit):
        qml.IsingXX(params[q], wires=[q, (q+1)%n_qubit])

def MølmerSørensenXXCircuitModel(n_qubit, circuit_depth):
    
    dev = qml.device("default.qubit", wires=n_qubit)
    
    @qml.qnode(dev, interface="torch", diff_method="backprop")    
    def circuit(params, su2_params):
        
        SU2_rot_layer(su2_params, n_qubit)
        for l in range(circuit_depth):
            MølmerSørensenXXLayer(params[0, :, l: l+1], n_qubit)
            RY_rot_layer(params[1, :, l: l+1], n_qubit)

        return qml.probs(wires=list(range(n_qubit)))

    return circuit

