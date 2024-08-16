import pennylane as qml
from pennylane import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils import (
    epsilon, evaluate, mutual_information, entanglement_of_formation
)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from functools import partial
from pprint import pprint
from data import *
import json

def operator_pool(n_qubit, selected_pairs=None):

    pool = []
    gate_description = []
    if selected_pairs:
        for i, j in selected_pairs:
            pool.append(partial(PauliStringRotation, pauli_string='XY', qubit=[i, j]))
            pool.append(partial(PauliStringRotation, pauli_string='XY', qubit=[j, i]))
            pool.append(partial(PauliStringRotation, pauli_string='YZ', qubit=[i, j]))
            pool.append(partial(PauliStringRotation, pauli_string='YZ', qubit=[j, i]))
            # pool.append(partial(qml.CRY, wires=[i, j]))
            # pool.append(partial(qml.CRY, wires=[j, i]))
            gate_description.append(f'e^[X{i} Y{j}]')
            gate_description.append(f'e^[X{j} Y{i}]')
            gate_description.append(f'e^[Y{i} Z{j}]')
            gate_description.append(f'e^[Y{j} Z{i}]')
            # gate_description.append(f'CRY[{i}, {j}]')
            # gate_description.append(f'CRY[{j}, {i}]')

        for i in range(n_qubit):
            pool.append(partial(qml.RY, wires=i))
            gate_description.append(f'RY[{i}]')

    else:
        for i in range(n_qubit):
            for j in range(n_qubit):
                if i != j:
                    pool.append(partial(PauliStringRotation, pauli_string='XY', qubit=[i, j]))
                    pool.append(partial(PauliStringRotation, pauli_string='YZ', qubit=[i, j]))
                    # pool.append(partial(qml.CRY, wires=[i, j]))
                    gate_description.append(f'e^[X{i} Y{j}]')
                    gate_description.append(f'e^[Y{i} Z{j}]')
                    # gate_description.append(f'CRY[{i}, {j}]')
        for i in range(n_qubit):
            pool.append(partial(qml.RY, wires=i))
            gate_description.append(f'RY[{i}]')
        
    return pool, gate_description

def entanglement_measure(target_prob, n_qubit):

    state = torch.sqrt(target_prob)
    ent_list = []
    
    for i in range(n_qubit):
        for j in range(i+1, n_qubit):
            MI = mutual_information(state, subsystems=(i, j))
            EOF = entanglement_of_formation(state, subsystems=(i, j))
            ent_list.append([(i, j), MI, EOF])

    ent_list = sorted(ent_list, key=lambda x: x[1], reverse=True)

    return ent_list

def select_qubit_pairs(target_prob, n_qubit, reduction_rate=0.5):

    ent_list = entanglement_measure(target_prob, n_qubit)    
    pair_list = [x[0] for x in ent_list]
    MI_list = np.array([x[1] for x in ent_list])
    MI_max = MI_list[1]
    N_selected_pairs = np.sum(MI_list > MI_max * reduction_rate)
    return pair_list[:N_selected_pairs]
            
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

class ACLBM:

    def __init__(self, 
                 data_class: LogNormal | Triangular | Bimodal | 
                             BarAndStripes | RealImage,
                 n_epoch: int,
                 n_iter: int,
                 No: int,
                 alpha: float,
                 reduction_rate=None
                 ):
        
        self.data_class = data_class
        self.n_qubit = data_class.n_bit
        self.n_epoch = n_epoch
        self.n_iter = n_iter
        self.threshold1 = 5e-3
        self.threshold2 = 1e-2
        self.No = No
        self.alpha = alpha
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.target_prob = torch.Tensor(data_class.get_data()).double().to(self.device)
        
        if reduction_rate is not None:
            selected_pairs = select_qubit_pairs(self.target_prob, self.n_qubit, reduction_rate=reduction_rate)
            self.pool, self.gate_description = operator_pool(self.n_qubit, selected_pairs)
        else:
            self.pool, self.gate_description = operator_pool(self.n_qubit)

        self.params = nn.ParameterDict({
            'Ry-layer': nn.Parameter(torch.full((self.n_qubit,), np.pi/2), requires_grad=True),
            'Append': nn.Parameter(torch.Tensor([]), requires_grad=True)
        }).to(self.device)

        self.operatorID = []
        self.loss_history = {
            'epoch': [],
            'iteration': []
        }
        self.kl_history = {
            'epoch': [],
            'iteration': []
        }
        self.js_history = {
            'epoch': [],
            'iteration': []
        }
        self.grad_norm_history = []

        Criterion = {
            'KL divergence': lambda p, q: torch.inner(p[p>0], torch.log(p[p>0] / q[p>0])),
            'Renyi-0.5 divergence': lambda p, q: -2 * torch.log(torch.sum(torch.sqrt(p[p>0] * q[p>0]))),
            'Renyi-2 divergence': lambda p, q: torch.log(torch.sum(p[p>0] ** 2 / (q[p>0]))),
            'log-MSE': lambda p, q: torch.log(torch.sum((p-q)**2)),
            'Fisher-Rao metric': lambda p, q: torch.arccos(torch.inner(torch.sqrt(p), torch.sqrt(q)))
        }

        # self.criterion = Criterion['Fisher-Rao metric']
        self.criterion = Criterion['KL divergence'] 
        # self.criterion = Criterion['log-MSE']

        if reduction_rate is not None:
            self.filename = f'./images/ACLBM/ACLBM-Fisher-Rao(data={data_class.name}, No={self.No}, t1={self.threshold1}, t2={self.threshold2}, r={reduction_rate}).png'
            self.result_file = f'./results/ACLBM/ACLBM-Fisher-Rao(data={data_class.name}, No={self.No}, t1={self.threshold1}, t2={self.threshold2}, r={reduction_rate}).json'
        else:
            self.filename = f'./images/ACLBM/ACLBM-Fisher-Rao(data={data_class.name}, No={self.No}, t1={self.threshold1}, t2={self.threshold2}).png'
            self.result_file = f'./results/ACLBM/ACLBM-Fisher-Rao(data={data_class.name}, No={self.No}, t1={self.threshold1}, t2={self.threshold2}).json'

    def circuit(self, ry_params, append_params, eval_params=None, mode=None):

        if mode == 'Train':

            for q in range(self.n_qubit):
                qml.RY(ry_params[q], wires=q)

            for i, id in enumerate(self.operatorID):
                gate = self.pool[id]
                gate(append_params[i])
            
            return qml.probs(wires=list(range(self.n_qubit)))

        elif mode == 'Gradient-evaluation':

            for q in range(self.n_qubit):
                qml.RY(ry_params[q].detach(), wires=q)

            for i, id in enumerate(self.operatorID):
                gate = self.pool[id]
                gate(append_params[i].detach())

            for i, gate in enumerate(self.pool):
                gate(eval_params[i])
            
            return qml.probs(wires=list(range(self.n_qubit)))

    def select_operator(self):
        
        dev = qml.device('default.qubit.torch', wires=self.n_qubit)
        
        circuit = self.circuit
        eval_params = nn.Parameter(torch.zeros(len(self.pool)), requires_grad=True)
        model = qml.QNode(circuit, dev, interface='torch', diff_method='backprop')
        prob = model(self.params['Ry-layer'], self.params['Append'], eval_params, mode='Gradient-evaluation')
        loss = self.criterion(self.target_prob, prob)
        loss.backward()
        grads = eval_params.grad.detach().cpu().numpy()
        grads = np.abs(grads)
        if len(grads) > self.No:
            selected_index = np.argsort(grads)[::-1][:self.No].tolist()
        else:
            selected_index = np.argsort(grads)[::-1].tolist()
        selected_gate = [self.gate_description[index] for index in selected_index]
        max_grad = [grads[index] for index in selected_index]

        return max_grad, selected_index, selected_gate
    
    def plot_training_result(self, prob):
        
        fig = plt.figure(figsize=(10, 9))
        gs = gridspec.GridSpec(2, 1, figure=fig)
 
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])
        
        n_epoch = len(self.kl_history['epoch'])

        ax1.semilogy(np.arange(n_epoch)+1, self.kl_history['epoch'], label='KL divergence', color='red')
        ax1.semilogy(np.arange(n_epoch)+1, self.js_history['epoch'], label='JS divergence', color='blue')
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('KL / JS divergence')
        ax1.grid()
        ax1.legend(loc='upper right')

        if isinstance(self.data_class, RealImage):
            if self.data_class.remapped:
                ax2.imshow(prob.detach().cpu().numpy()[self.data_class.inverse_indices].reshape(256, 256))
            else:
                ax2.imshow(prob.detach().cpu().numpy().reshape(256, 256))
            
        else:
            ax2.bar(np.arange(prob.shape[0])+1, self.target_prob.detach().cpu().numpy(), alpha=0.5, color='blue', label='target')
            ax2.bar(np.arange(prob.shape[0])+1, prob.detach().cpu().numpy(), alpha=0.5, color='red', label='approx')
            ax2.legend(loc='upper right')

        plt.savefig(self.filename)
    
    def fit(self):
        
        dev = qml.device('default.qubit.torch', wires=self.n_qubit)
        circuit = self.circuit
        model = qml.QNode(circuit, dev, interface='torch', diff_method='backprop')

        for i_iter in range(self.n_iter):

            max_grad, max_grad_index, max_grad_gate = self.select_operator()

            pprint(f'==== Found maximium gradient {max_grad} of gate ' + ', '.join(max_grad_gate) + ' ====')
            
            if max_grad[0] < self.threshold1:
                print('Convergence criterion has reached, break the loop!')
                break

            self.operatorID += max_grad_index
            self.params['Append'] = torch.cat((self.params['Append'], nn.Parameter(torch.zeros(len(max_grad)), requires_grad=True).to(self.device)))
            lr = torch.linalg.vector_norm(torch.Tensor(max_grad)).item() / np.sqrt(self.No) * self.alpha
            print('learning rate = ', lr)
            
            if i_iter == 0:
                opt = optim.Adam(self.params.values(), lr=lr)
            else:
                opt.param_groups[0]['lr'] = lr
                opt.param_groups[0]['params'][0] = self.params['Append']

            while True:
                opt.zero_grad()
                prob =  model(self.params['Ry-layer'], self.params['Append'], mode='Train')
                loss = self.criterion(self.target_prob, prob)
                loss.backward()
                opt.step()
                self.loss_history['epoch'].append(loss.item())
                kl_div, js_div = evaluate(self.target_prob, prob)
                self.kl_history['epoch'].append(kl_div)
                self.js_history['epoch'].append(js_div)

                grad_vec = torch.cat((self.params['Append'].grad, self.params['Ry-layer'].grad))
                grad_norm = torch.linalg.vector_norm(grad_vec)
                print(kl_div, loss.item(), grad_norm.item())
                
                if grad_norm < self.threshold2:
                    break

            print(f"iteration: {i_iter+1} | epoch: {len(self.loss_history['epoch'])+1} |   loss: {loss.item():.6f}  |   KL divergence: {kl_div:.6f}  |  JS divergence: {js_div:.6f}")
            self.loss_history['iteration'].append(loss.item())
            self.kl_history['iteration'].append(kl_div)
            self.js_history['iteration'].append(js_div)
            
            if len(self.loss_history['epoch']) + 1 >= self.n_epoch:
                break
            
        self.plot_training_result(prob)

        with open(self.result_file, 'w') as f:
            data_dict = {
                'pmf': prob.detach().cpu().tolist(),
                'kl div': self.kl_history,
                'js div': self.js_history,
                'loss history': self.loss_history,
                'grad norm': self.grad_norm_history,
                'operator ID': self.operatorID
            }
            json.dump(data_dict, f)

if __name__ == '__main__':

    model = ACLBM(
        data_class=DATA_HUB['real image 1 (R)'],
        n_epoch=12000,
        n_iter=250,
        No=3,
        alpha=0.01,
        reduction_rate=None
    )
    
    model.fit()