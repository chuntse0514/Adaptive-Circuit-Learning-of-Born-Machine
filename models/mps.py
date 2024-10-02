import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from functools import partial
from pprint import pprint
from data import *
import json

def TowLocalPauliRotation(theta, pauli_string: str, qubits: list):
    
    if 'I' in pauli_string:
        for pauli, qindex in zip(pauli_string, qubits):
            if pauli == 'X':
                qml.RX(theta, wires=qindex)
            elif pauli == 'Y':
                qml.RY(theta, wires=qindex)
            elif pauli == 'Z':
                qml.RZ(theta, wires=qindex)
    else:
        for pauli, qindex in zip(pauli_string, qubits):
            if pauli == 'X':
                qml.RY(-np.pi/2, wires=qindex)
            elif pauli == 'Y':
                qml.RX(np.pi/2, wires=qindex)

        qml.CNOT(wires=qubits)
        qml.RZ(theta, wires=qubits[1])
        qml.CNOT(wires=qubits)
        
        for pauli, qindex in zip(pauli_string, qubits):
            if pauli == 'X':
                qml.RY(-np.pi/2, wires=qindex)
            elif pauli == 'Y':
                qml.RX(np.pi/2, wires=qindex)

def General_SU2_Rotation(params, qubits: list):
    
    # SU(2) x SU(2) rotation
    qml.RZ(params[0], wires=qubits[0])
    qml.RY(params[1], wires=qubits[0])
    qml.RZ(params[2], wires=qubits[0])
    
    qml.RZ(params[3], wires=qubits[1])
    qml.RY(params[4], wires=qubits[1])
    qml.RZ(params[5], wires=qubits[1])
    
    # XX rotation
    qml.RY(-np.pi/2, wires=qubits[0])
    qml.RY(-np.pi/2, wires=qubits[1])
    qml.CNOT(wires=qubits)
    qml.RZ(params[6], wires=qubits[1])
    qml.CNOT(wires=qubits)
    qml.RY(np.pi/2, wires=qubits[0])
    qml.RY(np.pi/2, wires=qubits[1])
    
    ## YY rotation
    qml.RX(np.pi/2, wires=qubits[0])
    qml.RX(np.pi/2, wires=qubits[1])
    qml.CNOT(wires=qubits)
    qml.RZ(params[7], wires=qubits[1])
    qml.CNOT(wires=qubits)
    qml.RX(-np.pi/2, wires=qubits[0])
    qml.RX(-np.pi/2, wires=qubits[1])
    
    ## ZZ rotation
    qml.CNOT(wires=qubits)
    qml.RZ(params[8], wires=qubits[1])
    qml.CNOT(wires=qubits)
    
    # SU(2) x SU(2) rotation
    qml.RZ(params[9], wires=qubits[0])
    qml.RY(params[10], wires=qubits[0])
    qml.RZ(params[11], wires=qubits[0])
    
    qml.RZ(params[12], wires=qubits[1])
    qml.RY(params[13], wires=qubits[1])
    qml.RZ(params[14], wires=qubits[1])
    


class MPS(nn.Module):
    
    def __init__(self, n_qubit: int, k: int):
        super().__init__()
        
        dev = qml.device('default.qubit.torch', wires=n_qubit)
        
        self.params = nn.ParameterList(
            # [nn.Parameter((torch.rand((n_qubit-1) * 15) * 2 - 1) * np.pi, requires_grad=True) for _ in range(k)] # uniform initialization
            [nn.Parameter(torch.normal(mean=torch.zeros((n_qubit-1) * 15), std=torch.Tensor([np.pi/8])), requires_grad=True) for _ in range(k)] # normal initialization
        )
        
        @qml.qnode(dev, interface='torch', diff_method='backprop')
        def circuit():
            for layer in range(k):
                for q in range(n_qubit-1):
                    for i, pauli_string in enumerate(['IX', 'IY', 'IZ', 
                                                      'XI', 'XX', 'XY', 'XZ',
                                                      'YI', 'YX', 'YY', 'YZ', 
                                                      'ZI', 'ZX', 'ZY', 'ZZ']):
                        TowLocalPauliRotation(self.params[layer][i], pauli_string, [q, q+1])
                    # General_SU2_Rotation(self.params[layer], [q, q+1])
                        
            return qml.probs()
        
        self.model = circuit
        
    def forward(self):
        return self.model()
                            
                            

class TensorNetwork:
    
    def __init__(self, 
                 data_class: LogNormal | Triangular | Bimodal | 
                             BarAndStripes | RealImage,
                 n_epoch: int,
                 reps: int,
                 lr: float,
                 ):
        self.data_class = data_class
        self.n_qubit = data_class.n_bit
        self.n_epoch = n_epoch
        self.reps = reps
        self.lr = lr
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu' 
        self.target_prob = torch.Tensor(data_class.get_data()).double().to(self.device)
        
        self.MPS = MPS(self.n_qubit, k=reps).to(self.device)
        self.optim = optim.Adam(params=self.MPS.parameters(), lr=lr, amsgrad=True)
        self.kl_div_fn = lambda p, q: torch.inner(p[p>0], torch.log(p[p>0] / q[p>0]))
        self.js_div_fn = lambda p, q: 1 / 2 * self.kl_div_fn(p, (p+q)/2) + 1 / 2 * self.kl_div_fn(q, (p+q)/2)
        
        self.loss_history = []
        self.kl_history = []
        self.js_history = []
        self.grad_norm_history = []
        
        self.filename = f'./images/MPS/MPS-1(data={data_class.name}, lr={lr}, reps={reps}).png'
        self.result_file = f'./results/MPS/MPS-1(data={data_class.name}, lr={lr}, reps={reps}).json'
        
    def plot_training_result(self, prob):
        
        fig = plt.figure(figsize=(10, 9))
        gs = gridspec.GridSpec(2, 1, figure=fig)
 
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])
        
        n_epoch = len(self.kl_history)

        ax1.semilogy(np.arange(n_epoch)+1, self.kl_history, label='KL divergence', color='red')
        ax1.semilogy(np.arange(n_epoch)+1, self.js_history, label='JS divergence', color='blue')
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('KL / JS divergence')
        ax1.grid()
        ax1.legend()

        if isinstance(self.data_class, RealImage):
            ax2.imshow(prob.detach().cpu().numpy().reshape(256, 256))
            
        else:
            ax2.bar(np.arange(prob.shape[0])+1, self.target_prob.detach().cpu().numpy(), alpha=0.5, color='blue', label='target')
            ax2.bar(np.arange(prob.shape[0])+1, prob.detach().cpu().numpy(), alpha=0.5, color='red', label='approx')
            ax2.legend()

        plt.savefig(self.filename)
        
    def fit(self):
        
        for i_epoch in range(self.n_epoch):
            prob = self.MPS()
            kl_div = self.kl_div_fn(self.target_prob, prob)
            js_div = self.js_div_fn(self.target_prob, prob)

            self.optim.zero_grad()
            kl_div.backward()
            self.optim.step()
            self.loss_history.append(kl_div.item())
            self.kl_history.append(kl_div.item())
            self.js_history.append(js_div.item())

            grad_vec = torch.cat([params.grad for params in self.MPS.params])
            grad_norm = torch.linalg.vector_norm(grad_vec)
            
            self.grad_norm_history.append(grad_norm.item())

            print(kl_div.item(), grad_norm.item())
            
            if (i_epoch + 1) % 5 == 0:
                print(f'epoch: {i_epoch+1} | KL divergence: {kl_div.item():6f} | JS divergence: {js_div.item():6f}')

            if grad_norm < 1e-2:
                break
            
        self.plot_training_result(prob)

        with open(self.result_file, 'w') as f:
            data_dict = {
                'pmf': prob.detach().cpu().tolist(),
                'kl div': self.kl_history,
                'js div': self.js_history,
                'loss history': self.loss_history,
                'grad norm': self.grad_norm_history
            }
            json.dump(data_dict, f)
            
            
if __name__ == "__main__":
    
    TN = TensorNetwork(
        data_class=DATA_HUB['real image 1_1'],
        n_epoch=8000,
        reps=3,
        lr=0.01
    )
    
    TN.fit()
