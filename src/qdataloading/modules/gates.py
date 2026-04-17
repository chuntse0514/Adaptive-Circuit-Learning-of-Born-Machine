import pennylane as qml
from pennylane import numpy as np

def PauliStringRotation(theta, pauli_string: str, qubit: list[int]):
    # Basis rotation
    for pauli, qindex in zip(pauli_string, qubit):
        if pauli == 'X':
            qml.RY(-np.pi / 2, wires=qindex)
        elif pauli == 'Y':
            qml.RX(np.pi / 2, wires=qindex)
    
    # CNOT layer
    for q, q_next in zip(qubit[:-1], qubit[1:]):
        qml.CNOT(wires=[q, q_next])
    
    # Z rotation
    qml.RZ(theta, wires=qubit[-1])

    # CNOT layer
    for q, q_next in zip(reversed(qubit[:-1]), reversed(qubit[1:])):
        qml.CNOT(wires=[q, q_next])

    # Basis rotation
    for pauli, qindex in zip(pauli_string, qubit):
        if pauli == 'X':
            qml.RY(np.pi / 2, wires=qindex)
        elif pauli == 'Y':
            qml.RX(-np.pi / 2, wires=qindex)
