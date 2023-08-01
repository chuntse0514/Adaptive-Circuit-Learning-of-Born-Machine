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

def entangleBlock(params, q_tuple):

    q1, q2 = q_tuple

    qml.CRY(params[0], wires=[q1, q2])
    qml.IsingXY(params[1], wires=[q1, q2])


def TwoQubitEntangleCircuitModel(n_qubit):
    
    counter = 0

    def TwoQubitEntangleLayer(params, q_tuple):
        nonlocal counter
        new_q_tuples, q_configs = split(q_tuple)
        q_center = (q_tuple[0] + q_tuple[1]) // 2
        if len(new_q_tuples) == 0:
            return
        
        for new_q_tuple, q_config in zip(new_q_tuples, q_configs):
            if q_config == 'normal':
                for q in range(*new_q_tuple):
                    entangleBlock(params[counter], q_tuple=(q, q+1))
                    counter += 1
            
            elif q_config == 'reverse':
                for q in reversed(range(*new_q_tuple)):
                    entangleBlock(params[counter], q_tuple=(q+1, q))
                    counter += 1
                entangleBlock(params[counter], q_tuple=(q_center, q_center+1))
                counter += 1

        for new_q_tuple in new_q_tuples:
            TwoQubitEntangleLayer(params, new_q_tuple)
    
    dev = qml.device('default.qubit.torch', wires=n_qubit)
    @qml.qnode(dev, interface='torch', diff_method='backprop')
    def circuit(params, su2_params):
        nonlocal counter
        counter = 0
        if n_qubit == 2:
            entangleBlock(params[0], q_tuple=(0, 1))
            counter = 1
        if n_qubit == 3:
            entangleBlock(params[0], q_tuple=(0, 1))
            entangleBlock(params[1], q_tuple=(1, 2))
            counter = 2
        else:
            TwoQubitEntangleLayer(params, (0, n_qubit-1))
        SU2_rot_layer(su2_params, n_qubit)
        return qml.probs(wires=range(n_qubit))

    @qml.qfunc_transform
    def reverse_circuit(q_script):
        for operation in reversed(q_script.operations):
            qml.apply(operation)
        qml.probs(wires=range(n_qubit))

    circuit.func = reverse_circuit(circuit.func)

    return circuit

def HardwareEfficientModel(n_qubit, circuit_depth):

    dev = qml.device('default.qubit.torch', wires=n_qubit)
    @qml.qnode(dev, interface='torch', diff_method='backprop')
    def circuit(ent_params, rot_params, su2_params):
        SU2_rot_layer(su2_params, n_qubit)
        for k in range(circuit_depth):
            for q in range(n_qubit-1):
                entangleBlock(ent_params[q, k, :], q_tuple=(q, q+1))
            RY_rot_layer(rot_params[:, k], n_qubit)

        return qml.probs(wires=range(n_qubit))
    return circuit

def split(q_tuple):
    new_q_tuples = []
    q_configs = []
    q_start, q_end = q_tuple
    if (q_end + q_start) // 2 - q_start > 0:
        new_q_tuples.append((q_start, (q_end + q_start) // 2))
        q_configs.append('normal')
    if q_end - (q_end + q_start) // 2 - 1 > 0:
        new_q_tuples.append(((q_end + q_start) // 2 + 1, q_end))
        q_configs.append('reverse')

    return new_q_tuples, q_configs

def countTotalParameter(n_qubit):
    counter = 0
    
    if n_qubit == 2:
        return 1
    elif n_qubit == 3:
        return 2

    def counterFunction(q_tuple):
        nonlocal counter
        new_q_tuples, q_configs = split(q_tuple)
        for new_q_tuple, q_config in zip(new_q_tuples, q_configs):
            counter += new_q_tuple[1] - new_q_tuple[0]
            if q_config == 'reverse':
                counter += 1
        for new_q_tuple in new_q_tuples:
            counterFunction(new_q_tuple)
    
    counterFunction((0, n_qubit-1))

    return counter